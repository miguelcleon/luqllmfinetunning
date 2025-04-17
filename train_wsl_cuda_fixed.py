import torch
import os
import sys
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import datasets

# Force garbage collection at start
gc.collect()
torch.cuda.empty_cache()

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    print(f"Current CUDA device: {current_device} - {torch.cuda.get_device_name(current_device)}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(current_device) / 1024 ** 2:.2f} MB")
    print(f"CUDA memory reserved: {torch.cuda.memory_reserved(current_device) / 1024 ** 2:.2f} MB")

    # Run a simple CUDA operation to verify functionality
    try:
        x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        y = x + x
        print(f"CUDA test successful: {y}")
        del x, y  # Clean up right away
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"CUDA test failed: {e}")

# Load a small sample of the dataset first to save memory
dataset_dict = datasets.load_dataset(
    "json",
    data_files="./trainingdata/combined_publications.json"
)

# Check dataset format
dataset = dataset_dict['train']
print(f"Dataset loaded with {len(dataset)} examples")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
tokenizer.pad_token = tokenizer.eos_token

# Configure quantization for GPU only (no CPU offload)
if torch.cuda.is_available():
    print("Setting up 4-bit quantization (GPU only)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )
    device_map = "auto"  # Keep everything on GPU
else:
    print("WARNING: No GPU available, defaulting to CPU (very slow)")
    bnb_config = None
    device_map = "cpu"

# Free memory
gc.collect()
torch.cuda.empty_cache()

# Load model
try:
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b-it",
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map=device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Prepare the model for training with kbit
# This is critical for making gradient checkpointing work with quantized models
if torch.cuda.is_available():
    print("Preparing model for kbit training...")
    model = prepare_model_for_kbit_training(model)
    print("Model prepared for kbit training")

# Configure LoRA with minimal parameters
lora_config = LoraConfig(
    r=4,  # Very small rank
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]  # Reduced target modules
)

# Apply LoRA to model
peft_model = get_peft_model(model, lora_config)
print("Model prepared with LoRA")

# Free memory
gc.collect()
torch.cuda.empty_cache()


# Prepare dataset for instruction tuning
def prepare_instruction_dataset(examples):
    # Format as instruction tuning
    prompts = []
    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        input_text = examples['input'][i] if examples['input'][i] else ""
        output = examples['output'][i]

        # Format following Gemma instruction format
        if input_text:
            prompt = f"<start_of_turn>user\n{instruction}\n{input_text}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"
        else:
            prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"

        prompts.append(prompt)

    # Tokenize the prompts
    tokenized_inputs = tokenizer(prompts, padding=False, truncation=True, max_length=256)  # Very short context
    return tokenized_inputs


# Process and tokenize the dataset
tokenized_dataset = dataset.map(
    prepare_instruction_dataset,
    batched=True,
    remove_columns=['instruction', 'input', 'output']
)
print("Dataset tokenized")

# Set up training arguments with extremely conservative settings
training_args = TrainingArguments(
    output_dir="./trainingdata",
    num_train_epochs=5,  # Just one epoch for testing
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # Large accumulation to avoid OOM
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    logging_steps=1,  # Log frequently to see progress
    save_strategy="epoch",
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    # Disable gradient checkpointing initially - we'll enable it manually
    gradient_checkpointing=False,
    dataloader_num_workers=0,  # Disable parallel loading
    optim="adamw_torch",
    report_to="none",
    max_grad_norm=0.3,  # Limit gradient norms
)

# Explicitly set requires_grad on the LoRA parameters
for name, param in peft_model.named_parameters():
    if "lora" in name:
        param.requires_grad = True
        print(f"Parameter {name} requires_grad set to {param.requires_grad}")

# Enable gradient checkpointing manually after setting requires_grad
if hasattr(peft_model, "enable_input_require_grads"):
    peft_model.enable_input_require_grads()
else:
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)


    peft_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

# Enable gradient checkpointing manually
if hasattr(peft_model, "gradient_checkpointing_enable"):
    print("Enabling gradient checkpointing...")
    peft_model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled")

# Initialize the trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Free memory before training
gc.collect()
torch.cuda.empty_cache()

# Start training with error handling
print("Starting training...")
try:
    trainer.train()
except RuntimeError as e:
    if "out of memory" in str(e):
        print("===== CUDA OUT OF MEMORY ERROR =====")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
        print("\nTry these options:")
        print("1. Reduce context length further")
        print("2. Try CPU-only training")
        print("3. Reduce LoRA rank to 2")
        print("4. Use a smaller model")
        sys.exit(1)
    else:
        print(f"Training error: {e}")
        sys.exit(1)

# Save the fine-tuned model
peft_model.save_pretrained("./trainingdata/final_modelv2")
tokenizer.save_pretrained("./trainingdata/final_modelv2")
print(f"Training complete. Model saved to ./trainingdata/final_modelv2")