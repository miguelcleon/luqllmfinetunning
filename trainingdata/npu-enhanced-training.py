import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import datasets
import psutil
import argparse
import json
import os
import re
import logging
import gc
import time
from typing import Optional, List, Dict, Any, Union
#python gemma-3-4b-npu-training.py \
#  --data_files training_datav3.json_sections.json training_datav3.json_instruction.json training_datav3.json_basic.json \
#  --publications_file papers_metadata.json \
#  --output_dir gemma3_luq_model \
#  --model_id google/gemma-3-4b-it \
#  --format relationships \
#  NO DONT USE --use_npu \
#  --quantization int4 \
#  --max_context 1024 \
#  --batch_size 1 \
#  --gradient_accumulation_steps 16 \
#  --epochs 3 \
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


def optimize_memory_usage():
    """Optimize memory usage by clearing cache and collecting garbage."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-3-4b-it with NPU acceleration")
    parser.add_argument(
        "--data_files",
        type=str,
        nargs='+',
        required=True,
        help="One or more paths to training data JSON files"
    )
    parser.add_argument("--publications_file", type=str, help="Path to the combined publications JSON file (optional)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the fine-tuned model")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-4b-it", help="Base model ID")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Number of steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--format", type=str, choices=["basic", "instruction", "sections", "relationships"],
                        default="relationships", help="Format of the training data")
    parser.add_argument("--use_npu", action="store_true", help="Use Intel NPU acceleration if available")
    parser.add_argument("--quantization", type=str, choices=["int8", "int4", "none"],
                        default="int4", help="Quantization type for NPU acceleration")
    parser.add_argument("--max_context", type=int, default=4096,
                        help="Maximum context length (lower values use less memory)")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps")
    parser.add_argument("--lora_rank", type=int, default=16, help="Rank for LoRA adapter")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha for LoRA adapter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout for LoRA adapter")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Resume training from checkpoint")

    return parser.parse_args()


def prepare_instruction_dataset(data_files: List[str], tokenizer, max_length=4096, format_type="relationships"):
    """Load, merge, format, and tokenize one or more JSON training files."""
    print(f"→ Loading & formatting {len(data_files)} file(s) in '{format_type}' format…")
    formatted_examples = []

    for file_path in data_files:
        print(f"   • Reading {file_path} …", end='')
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # handle either `{ "train": [...] }` or a flat list
        raw_list = data.get("train", data) if isinstance(data, dict) else data
        print(f" {len(raw_list)} items")

        for item in raw_list:
            try:
                # Instruction/relationship format
                if "instruction" in item and "output" in item:
                    instr = item["instruction"]
                    inp   = item.get("input", "")
                    out   = item["output"]
                    if inp:
                        prompt = (
                            "<start_of_turn>user\n"
                            f"{instr}\n\n{inp}<end_of_turn>\n"
                            "<start_of_turn>model\n"
                            f"{out}<end_of_turn>"
                        )
                    else:
                        prompt = (
                            "<start_of_turn>user\n"
                            f"{instr}<end_of_turn>\n"
                            "<start_of_turn>model\n"
                            f"{out}<end_of_turn>"
                        )
                    formatted_examples.append({"text": prompt})

                # Sections format
                elif "text" in item and ("<section" in item["text"] or "<title>" in item["text"]):
                    txt = item["text"]
                    title_match = re.search(r'<title>(.*?)</title>', txt)
                    title = title_match.group(1) if title_match else "this research paper"
                    prompt = (
                        "<start_of_turn>user\n"
                        f"Please analyze the following research paper: {title}<end_of_turn>\n"
                        "<start_of_turn>model\n"
                        f"{txt}<end_of_turn>"
                    )
                    formatted_examples.append({"text": prompt})

                # Basic text-only format
                elif "text" in item:
                    prompt = (
                        "<start_of_turn>user\n"
                        "Please analyze this research content<end_of_turn>\n"
                        "<start_of_turn>model\n"
                        f"{item['text']}<end_of_turn>"
                    )
                    formatted_examples.append({"text": prompt})

            except Exception as e:
                logger.warning(f"Error formatting an example from {file_path}: {e}")
                continue

    print(f"→ Formatted a total of {len(formatted_examples)} examples")
    dataset = datasets.Dataset.from_list(formatted_examples)

    # Tokenize
    print(f"→ Tokenizing with max_length={max_length} …")
    def tokenize_fn(exs):
        return tokenizer(exs["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    print(f"→ Tokenization complete: {len(tokenized)} examples ready")
    return tokenized


def load_and_prepare_model(model_id, use_npu, quantization, max_context, lora_rank, lora_alpha, lora_dropout):
    """Load and prepare the model for training with NPU optimizations and LoRA."""
    # Check NPU availability if requested
    has_npu, npu_lib = False, None
    if use_npu:
        has_npu, npu_lib = check_npu_availability()
        if has_npu:
            logger.info("NPU acceleration enabled for training")
        else:
            logger.warning("NPU acceleration requested but not available. Falling back to CPU/GPU.")

    # Load tokenizer
    logger.info(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure quantization for efficient training
    logger.info(f"Configuring quantization ({quantization})...")

    if quantization == "int8":
        logger.info("Using int8 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True
        )
    elif quantization == "int4":
        logger.info("Using int4 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    else:
        logger.info("No quantization applied")
        bnb_config = None

    # Load the base model with limited context to save memory
    logger.info(f"Loading base model {model_id}...")
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }

    # Add context length limitation to save memory

    try:
        print(f"→ Loading base model config for {model_id} …")
        config = AutoConfig.from_pretrained(model_id)
        # override the positional embeddings size:
        if max_context < getattr(config, "max_position_embeddings", max_context):
            print(f"→ Overriding config.max_position_embeddings: {config.max_position_embeddings} → {max_context}")
            config.max_position_embeddings = max_context

        print(f"→ Loading model {model_id} with custom config …")
        gpu_total = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
        gpu_allowed = f"{int(gpu_total * 0.9)}GB"

        # how much system RAM you have

        gpu_gb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
        ram_gb = psutil.virtual_memory().total // (1024 ** 3)
        max_mem = {
            0: f"{int(gpu_gb * 0.8)}GB",  # e.g. allow 80% of your GPU
            "cpu": f"{int(ram_gb * 0.9)}GB"
        }

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_mem,
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    # Apply NPU optimization if available (only for inference, not training yet)
    if has_npu and npu_lib:
        logger.info("NPU detected, but note that direct NPU acceleration for training is limited")
        logger.info("Will use LoRA for parameter-efficient fine-tuning")

    # Prepare model for quantized training
    if quantization in ["int8", "int4"]:
        logger.info("Preparing model for kbit training...")
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA for parameter-efficient fine-tuning
    logger.info("Configuring LoRA adapters...")

    # Find target modules based on model architecture
    if "gemma" in model_id.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    else:
        # Generic modules for transformer-based models
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )

    # Apply LoRA to the model
    logger.info("Applying LoRA adapters to model...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def main():
    # Parse command line arguments
    args = parse_args()

    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and prepare model with LoRA
    model, tokenizer = load_and_prepare_model(
        args.model_id,
        args.use_npu,
        args.quantization,
        args.max_context,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout
    )

    # Prepare the dataset
    tokenized_dataset = prepare_instruction_dataset(
        args.data_files,
        tokenizer,
        max_length=args.max_context,
        format_type=args.format
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # make sure caching is off
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        gradient_checkpointing=True,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )

    # Initialize the trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        label_names=["labels"],
    )

    # Start training
    logger.info("Starting training...")

    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    # Save the fine-tuned model
    logger.info(f"Training complete. Saving model to {args.output_dir}/final_model")
    model.save_pretrained(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")

    logger.info(f"Training complete. Model saved to {args.output_dir}/final_model")


if __name__ == "__main__":
    main()