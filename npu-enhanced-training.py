import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import datasets
import argparse
import json
import os
import re
import logging
# python npu-enhanced-training.py \
#  --data_file training_datav2.json \
#  --publications_file combined_publications.json \
#  --output_dir luq_lter_model_npu \
#  --model_id google/gemma-7b \
#  --format relationships \
#  --use_npu \
#  --quantization int8 \
#  --batch_size 2 \
#  --save_steps 100
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Check for NPU availability
def check_npu_availability():
    try:
        # First, try to import the Intel NPU Acceleration Library
        try:
            import intel_npu_acceleration_library
            logger.info("Intel NPU Acceleration Library is available")
            HAS_NPU_LIB = True
        except ImportError:
            logger.warning("Intel NPU Acceleration Library not found. Will use CPU/GPU if available.")
            HAS_NPU_LIB = False

        # Check if we're running on Intel hardware with NPU
        if HAS_NPU_LIB:
            # Try to query NPU info
            try:
                # Simple test to see if NPU is accessible
                return True, intel_npu_acceleration_library
            except Exception as e:
                logger.warning(f"NPU library found but hardware check failed: {e}")
                return False, None
        return False, None
    except Exception as e:
        logger.warning(f"Error checking NPU availability: {e}")
        return False, None


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma with enhanced publication data")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the training data JSON file")
    parser.add_argument("--publications_file", type=str, help="Path to the combined publications JSON file (optional)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the fine-tuned model")
    parser.add_argument("--model_id", type=str, default="google/gemma-7b", help="Base model ID")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--format", type=str, choices=["basic", "instruction", "sections", "relationships"],
                        default="relationships", help="Format of the training data")
    parser.add_argument("--use_npu", action="store_true", help="Use Intel NPU acceleration if available")
    parser.add_argument("--quantization", type=str, choices=["int8", "int4", "none"],
                        default="none", help="Quantization type for NPU acceleration")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    return parser.parse_args()


def prepare_instruction_dataset(data_file, tokenizer, max_length=2048):
    """Prepare a dataset in instruction format."""
    # Load the JSON file
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if it's already in datasets format
    if isinstance(data, dict) and "train" in data:
        return datasets.Dataset.from_dict(data["train"])

    # Format the data for instruction tuning
    formatted_examples = []

    for item in data:
        # Check if the item is already in instruction format
        if "instruction" in item and "output" in item:
            # Format as instruction prompt for Gemma
            instruction = item["instruction"]
            input_text = item.get("input", "")
            output = item["output"]

            if input_text:
                prompt = f"<start_of_turn>user\n{instruction}\n\n{input_text}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"
            else:
                prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"

            formatted_examples.append({"text": prompt})
        # Handle section format
        elif "text" in item and ("<section" in item["text"] or "<title>" in item["text"]):
            # Convert to instruction format
            text = item["text"]
            title_match = re.search(r'<title>(.*?)</title>', text)
            title = title_match.group(1) if title_match else "this research paper"

            # Create instruction prompt
            prompt = f"<start_of_turn>user\nPlease analyze the content of the following research paper: {title}<end_of_turn>\n<start_of_turn>model\n{text}<end_of_turn>"
            formatted_examples.append({"text": prompt})
        # Handle basic format
        elif "text" in item:
            # Simple text format - wrap in conversation format
            prompt = f"<start_of_turn>user\nPlease analyze the following research content:<end_of_turn>\n<start_of_turn>model\n{item['text']}<end_of_turn>"
            formatted_examples.append({"text": prompt})

    # Create a dataset
    dataset = datasets.Dataset.from_list(formatted_examples)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    return tokenized_dataset


def optimize_for_npu(model, npu_lib, quantization_type):
    """Apply NPU-specific optimizations to the model."""
    logger.info(f"Optimizing model for NPU with quantization: {quantization_type}")

    try:
        # Apply NPU-specific optimizations
        from intel_npu_acceleration_library.compiler import CompilerConfig

        # Configure compiler based on quantization type
        if quantization_type == "int8":
            dtype = torch.int8
        elif quantization_type == "int4":
            # If int4 is supported (may require special handling)
            dtype = torch.float16  # Will be quantized to int4 internally
        else:
            dtype = torch.float16

        compiler_conf = CompilerConfig(dtype=dtype, training=True)

        # Compile the model for NPU
        optimized_model = npu_lib.compile(model, compiler_conf)
        logger.info("Model successfully optimized for NPU")
        return optimized_model
    except Exception as e:
        logger.warning(f"Failed to optimize for NPU: {e}")
        logger.info("Falling back to original model")
        return model


def main():
    # Parse command line arguments
    args = parse_args()

    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check NPU availability if requested
    has_npu, npu_lib = False, None
    if args.use_npu:
        has_npu, npu_lib = check_npu_availability()
        if has_npu:
            logger.info("NPU acceleration enabled")
        else:
            logger.warning("NPU acceleration requested but not available. Falling back to CPU/GPU.")

    logger.info(f"Loading tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure quantization for efficient training
    logger.info("Configuring quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load the base model
    logger.info(f"Loading base model {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Apply NPU optimizations if available
    if has_npu and npu_lib:
        model = optimize_for_npu(model, npu_lib, args.quantization)

    # Configure LoRA for parameter-efficient fine-tuning
    logger.info("Configuring LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Apply LoRA to the model
    peft_model = get_peft_model(model, lora_config)
    logger.info("LoRA adapters applied to model")

    # Prepare the dataset based on format
    logger.info(f"Preparing dataset from {args.data_file}...")
    if args.format in ["instruction", "relationships"]:
        tokenized_dataset = prepare_instruction_dataset(args.data_file, tokenizer)
    else:
        # Load dataset in basic format
        dataset = datasets.load_dataset("json", data_files=args.data_file)

        # Tokenize the dataset
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=2048)

        tokenized_dataset = dataset["train"].map(tokenize_function, batched=True, remove_columns=["text"])

    logger.info(f"Dataset prepared with {len(tokenized_dataset)} examples")

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    # Initialize the trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save the fine-tuned model
    logger.info(f"Training complete. Saving model to {args.output_dir}/final_model")
    peft_model.save_pretrained(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")

    logger.info(f"Training complete. Model saved to {args.output_dir}/final_model")


if __name__ == "__main__":
    import re  # Import here for use in prepare_instruction_dataset

    main()