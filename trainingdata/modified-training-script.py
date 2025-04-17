import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import datasets
import argparse
import json
import os
# python modified-training-script.py \
#  --data_file training_datav2.json \
#  --publications_file combined_publications.json \
#  --output_dir luq_lter_model \
#  --model_id google/gemma-7b \
#  --format relationships \
#  --epochs 3

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


def main():
    # Parse command line arguments
    args = parse_args()

    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure quantization for efficient training
    print("Configuring quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load the base model
    print(f"Loading base model {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Configure LoRA for parameter-efficient fine-tuning
    print("Configuring LoRA adapters...")
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
    print("LoRA adapters applied to model")

    # Prepare the dataset based on format
    print(f"Preparing dataset from {args.data_file}...")
    if args.format in ["instruction", "relationships"]:
        tokenized_dataset = prepare_instruction_dataset(args.data_file, tokenizer)
    else:
        # Load dataset in basic format
        dataset = datasets.load_dataset("json", data_files=args.data_file)

        # Tokenize the dataset
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=2048)

        tokenized_dataset = dataset["train"].map(tokenize_function, batched=True, remove_columns=["text"])

    print(f"Dataset prepared with {len(tokenized_dataset)} examples")

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
        save_strategy="epoch",
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
    print("Starting training...")
    trainer.train()

    # Save the fine-tuned model
    print(f"Training complete. Saving model to {args.output_dir}/final_model")
    peft_model.save_pretrained(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")

    print(f"Training complete. Model saved to {args.output_dir}/final_model")


if __name__ == "__main__":
    import re  # Import here for use in prepare_instruction_dataset

    main()