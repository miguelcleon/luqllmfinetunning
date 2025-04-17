# C:\one\OneDrive - USNH\LUQ-LTER-Share\Zotero
# C:\one\OneDrive - USNH\LUQ-LTER-Share\publications
# /mnt/c/one/OneDrive - USNH/LUQ-LTER-Share/publications
# /mnt/c/one/OneDrive - USNH/LUQ-LTER-Share/Zotero

import os
import json
import argparse
import subprocess
import shutil
import re
from tqdm import tqdm


def setup_environment():
    """Install required packages for the workflow."""
    packages = [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "PyPDF2",
        "PyMuPDF",
        "nltk",
        "scikit-learn",
        "tqdm"
    ]

    print("Setting up environment...")
    for package in tqdm(packages):
        subprocess.run(["pip", "install", "-q", package], check=True)

    print("Environment setup complete.")


def validate_dataset(dataset_path, min_entries=10, min_tokens=1000):
    """Validate the dataset to ensure it's suitable for training."""
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        if len(dataset) < min_entries:
            print(f"Warning: Dataset has only {len(dataset)} entries. Recommended: at least {min_entries}.")
            return False

        # Check token counts (approximate)
        short_entries = 0
        for entry in dataset:
            text = entry.get("text", "")
            # Rough approximation of token count (words / 0.75)
            token_count = len(text.split()) / 0.75
            if token_count < min_tokens:
                short_entries += 1

        if short_entries > len(dataset) * 0.2:  # More than 20% are short
            print(
                f"Warning: {short_entries} entries ({short_entries / len(dataset) * 100:.1f}%) are shorter than recommended.")
            return False

        print(f"Dataset validation passed: {len(dataset)} entries")
        return True

    except Exception as e:
        print(f"Error validating dataset: {e}")
        return False


def prepare_training_script(dataset_path, output_dir, model_id="google/gemma-7b", epochs=3):
    """Create a training script for fine-tuning Gemma."""
    # Create the training script content with proper formatting
    script = '''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import datasets

# Load dataset
dataset = datasets.load_dataset("json", data_files="{0}")
print(f"Dataset loaded with {{len(dataset['train'])}} examples")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{1}")
tokenizer.pad_token = tokenizer.eos_token

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "{1}",
    quantization_config=bnb_config,
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Apply LoRA to model
peft_model = get_peft_model(model, lora_config)
print("Model prepared with LoRA")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=2048)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print("Dataset tokenized")

# Set up training arguments
training_args = TrainingArguments(
    output_dir="{2}",
    num_train_epochs={3},
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
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
    train_dataset=tokenized_dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Start training
print("Starting training...")
trainer.train()

# Save the fine-tuned model
peft_model.save_pretrained("{2}/final_model")
tokenizer.save_pretrained("{2}/final_model")
print(f"Training complete. Model saved to {2}/final_model")
'''.format(dataset_path, model_id, output_dir, epochs)

    # Save the script
    script_path = os.path.join(output_dir, "train_gemma.py")
    os.makedirs(output_dir, exist_ok=True)

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script)

    print(f"Training script created at {script_path}")
    return script_path


def prepare_inference_script(model_dir):
    """Create an inference script for testing the fine-tuned model."""
    script = '''
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load the fine-tuned model
model_path = "{0}/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# Create a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Function to generate text
def generate_text(prompt, max_length=500):
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    return result[0]['generated_text']

# Interactive loop
print("\\nFine-tuned Gemma Model - Test Interface")
print("Enter 'q' or 'quit' to exit")
print("-" * 50)

while True:
    prompt = input("\\nEnter a prompt: ")

    if prompt.lower() in ['q', 'quit', 'exit']:
        break

    generated = generate_text(prompt)
    print("\\nGenerated text:")
    print("-" * 50)
    print(generated)
    print("-" * 50)
'''.format(model_dir)

    # Save the script
    script_path = os.path.join(model_dir, "inference.py")

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script)

    print(f"Inference script created at {script_path}")
    return script_path


def create_pdf_processor_script(output_dir):
    """Create a PDF processor script that handles recursive directory processing."""
    script = '''
import os
import re
import json
import argparse
from tqdm import tqdm
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def extract_metadata(pdf_path):
    """Extract metadata from PDF."""
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata

        # Extract title from the first page if not in metadata
        if not metadata.get('title'):
            first_page = doc[0]
            text = first_page.get_text()
            # Simple heuristic: first line might be the title
            lines = text.strip().split('\\n')
            if lines and len(lines[0]) < 200:  # Reasonable title length
                metadata['title'] = lines[0].strip()

        # Try to extract abstract
        abstract = ""
        for page in doc[:2]:  # Check first two pages
            text = page.get_text()
            abstract_match = re.search(r'Abstract[:\\.\s]+(.*?)(?:\\n\\n|\\nIntroduction|\\n1\\.)', 
                                      text, re.DOTALL | re.IGNORECASE)
            if abstract_match:
                abstract = abstract_match.group(1).strip()
                break

        metadata['abstract'] = abstract

        doc.close()
        return metadata
    except Exception as e:
        print(f"Error extracting metadata from {pdf_path}: {e}")
        return {}


def segment_paper(text):
    """Segment paper into sections."""
    sections = {}

    # Try to identify common section headers
    section_pattern = re.compile(r'\\n(\\d+\\.?\\s+[A-Z][a-zA-Z\\s]+|\\b(?:Abstract|Introduction|Methods|Results|Discussion|Conclusion|References)\\b)[:\\.\s\\n]+', re.IGNORECASE)

    matches = list(section_pattern.finditer(text))

    if not matches:
        # If no sections found, return the whole text as "body"
        return {"body": text}

    # Extract each section
    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        start_pos = match.end()

        # Get end position (start of next section or end of text)
        if i < len(matches) - 1:
            end_pos = matches[i+1].start()
        else:
            end_pos = len(text)

        section_text = text[start_pos:end_pos].strip()
        sections[section_name] = section_text

    return sections


def clean_text(text):
    """Clean extracted text for better quality."""
    # Replace multiple newlines with a single newline
    text = re.sub(r'\\n{3,}', '\\n\\n', text)

    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)

    # Remove page numbers
    text = re.sub(r'\\n\\s*\\d+\\s*\\n', '\\n', text)

    # Remove form feed characters
    text = re.sub(r'\\f', '', text)

    # Fix hyphenated words at line breaks
    text = re.sub(r'(\\w+)-\\n(\\w+)', r'\\1\\2', text)

    # Handle common PDF artifacts
    text = re.sub(r'\\(cid:\\d+\\)', '', text)

    # Fix spacing after periods
    text = re.sub(r'\\.([A-Z])', r'. \\1', text)

    # Clean up citations
    text = re.sub(r'\\[\\d+(?:,\\s*\\d+)*\\]', '', text)
    text = re.sub(r'\\(\\w+ et al\\.,? \\d{4}[a-z]?\\)', '', text)

    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\\\(\\\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def process_pdf(pdf_path, include_metadata=True, include_sections=True):
    """Process a single PDF with advanced options."""
    try:
        # Basic text extraction with PyMuPDF
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\\n"
        doc.close()

        # Clean the text
        text = clean_text(text)

        # Extract metadata if requested
        metadata = {}
        if include_metadata:
            metadata = extract_metadata(pdf_path)

        # Segment into sections if requested
        sections = {}
        if include_sections:
            sections = segment_paper(text)

        # Create document entry
        document = {
            "text": text,
            "source": pdf_path,
            "length": len(text)
        }

        # Add metadata if available
        if metadata:
            document["metadata"] = metadata

        # Add sections if available and requested
        if sections and include_sections:
            document["sections"] = sections

        return document

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None


def find_pdf_files(dir_path):
    """Find all PDF files in a directory and its subdirectories recursively."""
    pdf_files = []
    print(f"Searching for PDF files in {dir_path} and its subdirectories...")

    try:
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))

        print(f"Found {len(pdf_files)} PDF files")
        return pdf_files
    except Exception as e:
        print(f"Error searching for PDFs in {dir_path}: {e}")
        return []


def process_directory(dir_path, output_file, min_length=500, include_metadata=True, include_sections=True):
    """Process all PDFs in a directory and save as a JSON file."""
    # Find all PDF files recursively
    pdf_files = find_pdf_files(dir_path)

    if not pdf_files:
        print(f"No PDF files found in {dir_path} and its subdirectories.")
        return []

    # Process each PDF
    print(f"Processing {len(pdf_files)} PDF files...")
    dataset = []

    for pdf_file in tqdm(pdf_files):
        document = process_pdf(pdf_file, include_metadata, include_sections)

        if document and document.get("text") and len(document["text"]) >= min_length:
            dataset.append(document)
        else:
            print(f"Skipping {pdf_file}: Text too short or extraction failed")

    if not dataset:
        print("No valid documents were processed.")
        return []

    # Save as JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(dataset)} PDFs successfully out of {len(pdf_files)} total")
    print(f"Output saved to {output_file}")

    return dataset


def create_training_json(input_json, output_json, format_type="basic"):
    """
    Convert processed JSON to training format.

    format_type options:
    - "basic": Simple text entries
    - "instruction": Q&A format for instruction tuning
    - "sections": Section-based format
    """
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            print(f"Warning: Input JSON file {input_json} is empty.")
            return []
    except Exception as e:
        print(f"Error reading input JSON {input_json}: {e}")
        return []

    training_data = []

    for item in data:
        if format_type == "basic":
            # Simple format with just text
            training_data.append({"text": item["text"]})

        elif format_type == "instruction":
            # Instruction tuning format
            # Extract abstract as question, rest as answer if available
            if "metadata" in item and "abstract" in item["metadata"] and item["metadata"]["abstract"]:
                abstract = item["metadata"]["abstract"]
                # Create Q&A pairs from abstract and content
                training_data.append({
                    "instruction": f"Explain the following research abstract in detail: {abstract}",
                    "input": "",
                    "output": item["text"]
                })

                # Add more instruction examples if paper has sections
                if "sections" in item:
                    for section_name, section_text in item["sections"].items():
                        if section_name.lower() not in ["abstract", "references"]:
                            training_data.append({
                                "instruction": f"What does the paper say about {section_name}?",
                                "input": abstract,
                                "output": section_text
                            })
            else:
                # Fallback if no abstract
                training_data.append({
                    "instruction": "Summarize this research paper",
                    "input": "",
                    "output": item["text"]
                })

        elif format_type == "sections":
            # Format with sections as separate entries
            if "sections" in item:
                for section_name, section_text in item["sections"].items():
                    training_data.append({
                        "text": f"<section>{section_name}</section>\\n{section_text}"
                    })
            else:
                # Fallback to basic if no sections
                training_data.append({"text": item["text"]})

    # Save training data
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    print(f"Created training data with {len(training_data)} entries in {format_type} format")
    print(f"Training data saved to {output_json}")

    return training_data


def main():
    parser = argparse.ArgumentParser(description='Process PDF research papers recursively from directories')
    parser.add_argument('--input', required=True, help='Input directory containing PDFs (will be searched recursively)')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--min_length', type=int, default=500, help='Minimum text length to include (chars)')
    parser.add_argument('--metadata', action='store_true', help='Extract and include metadata')
    parser.add_argument('--sections', action='store_true', help='Segment papers into sections')
    parser.add_argument('--training_output', help='Create training-ready JSON file')
    parser.add_argument('--format', choices=['basic', 'instruction', 'sections'], default='basic',
                        help='Training data format')

    args = parser.parse_args()

    # Process directory of PDFs
    if os.path.isdir(args.input):
        dataset = process_directory(
            args.input, 
            args.output, 
            args.min_length,
            args.metadata, 
            args.sections
        )

        # Create training data if requested
        if args.training_output and dataset:
            create_training_json(args.output, args.training_output, args.format)
    else:
        print(f"Error: {args.input} is not a valid directory")


if __name__ == "__main__":
    main()
'''

    # Save the script
    script_path = os.path.join(output_dir, "recursive_pdf_processor.py")
    os.makedirs(output_dir, exist_ok=True)

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script)

    print(f"PDF processor script created at {script_path}")
    return script_path


def process_pdf_directories(pdf_dirs, output_dir, format_type="basic"):
    """Process PDFs from multiple directories and combine into a single dataset."""
    # Create processing directory
    processing_dir = os.path.join(output_dir, "processing")
    os.makedirs(processing_dir, exist_ok=True)

    # Create the PDF processor script
    processor_script = create_pdf_processor_script(processing_dir)

    # Lists to store intermediate and final JSON files
    intermediate_jsons = []

    # Process each PDF directory separately
    for i, pdf_dir in enumerate(pdf_dirs):
        print(f"\nProcessing directory {i + 1}/{len(pdf_dirs)}: {pdf_dir}")

        # Generate paths for this directory
        dir_basename = os.path.basename(os.path.normpath(pdf_dir))
        # Clean up directory name for file naming
        safe_dir_name = re.sub(r'[^\w\-_]', '_', dir_basename)
        raw_json = os.path.join(processing_dir, f"raw_papers_{safe_dir_name}.json")
        training_json = os.path.join(processing_dir, f"training_data_{safe_dir_name}_{format_type}.json")

        # Check if directory exists
        if not os.path.isdir(pdf_dir):
            print(f"Warning: Directory {pdf_dir} does not exist or is not accessible. Skipping.")
            continue

        # Process PDFs in this directory
        try:
            subprocess.run([
                "python", processor_script,
                "--input", pdf_dir,
                "--output", raw_json,
                "--metadata",
                "--sections",
                "--training_output", training_json,
                "--format", format_type
            ], check=True)

            # Add to list of intermediate JSONs
            intermediate_jsons.append(training_json)
        except subprocess.CalledProcessError as e:
            print(f"Error processing directory {pdf_dir}: {e}")
            continue

    # Combine all intermediate JSONs into a single dataset
    combined_json_path = os.path.join(output_dir, f"combined_training_data_{format_type}.json")

    if not intermediate_jsons:
        print("Error: No PDF directories were successfully processed.")
        return None

    # Combine JSON files
    print(f"\nCombining {len(intermediate_jsons)} datasets...")
    combined_data = []

    for json_file in intermediate_jsons:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"  - Added {len(data)} entries from {json_file}")
                combined_data.extend(data)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    # Save combined dataset
    with open(combined_json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

    print(f"Combined dataset saved with {len(combined_data)} entries to {combined_json_path}")
    return combined_json_path


def main():
    parser = argparse.ArgumentParser(description='Complete workflow for fine-tuning Gemma with research papers')
    parser.add_argument('--pdf_dirs', nargs='+', help='Multiple directories containing PDF research papers')
    parser.add_argument('--json_path', help='Path to existing JSON dataset (if already processed)')
    parser.add_argument('--output_dir', required=True, help='Output directory for models and scripts')
    parser.add_argument('--model', default="google/gemma-7b", help='Gemma model ID to fine-tune')
    parser.add_argument('--setup', action='store_true', help='Set up the environment')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--format', choices=['basic', 'instruction', 'sections'], default='basic',
                        help='Training data format')
    parser.add_argument('--train', action='store_true', help='Run the training script')

    args = parser.parse_args()

    # Set up environment if requested
    if args.setup:
        setup_environment()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process PDFs if provided
    if args.pdf_dirs:
        args.json_path = process_pdf_directories(args.pdf_dirs, args.output_dir, args.format)

    # Validate the dataset
    if not args.json_path:
        print("Error: Either --pdf_dirs or --json_path must be provided")
        return

    validate_dataset(args.json_path)

    # Prepare training script
    script_path = prepare_training_script(
        args.json_path,
        args.output_dir,
        model_id=args.model,
        epochs=args.epochs
    )

    # Prepare inference script
    inference_path = prepare_inference_script(args.output_dir)

    # Run training if requested
    if args.train:
        print(f"\nStarting training with {args.model}...")
        subprocess.run(["python", script_path], check=True)
    else:
        print(f"\nTo start training, run: python {script_path}")

    print(f"After training, you can test your model with: python {inference_path}")


if __name__ == "__main__":
    main()