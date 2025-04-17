import json
import re
import argparse
import random
from pathlib import Path


def count_tokens(text):
    """Estimate the number of tokens in a text.
    This is a simple approximation - OpenAI's tokenizer is more complex."""
    # Rough estimate: 1 token is about 4 characters for English text
    return len(text) // 4


def split_text(text, max_length):
    """Split text into chunks of approximately max_length tokens."""
    # Find natural breakpoints - paragraphs
    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_chunk = ""
    current_length = 0

    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph)

        # If a single paragraph exceeds max_length, split it into sentences
        if paragraph_tokens > max_length:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                sentence_tokens = count_tokens(sentence)

                # If adding this sentence would exceed the limit, start a new chunk
                if current_length + sentence_tokens > max_length and current_length > 0:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                    current_length = sentence_tokens
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
                    current_length += sentence_tokens
        else:
            # If adding this paragraph would exceed the limit, start a new chunk
            if current_length + paragraph_tokens > max_length and current_length > 0:
                chunks.append(current_chunk)
                current_chunk = paragraph
                current_length = paragraph_tokens
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_length += paragraph_tokens

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def parse_training_data(text, max_tokens=4096):
    """Parse the training data in the specified format and extract data elements."""
    # Find all data entries enclosed in curly braces
    pattern = r'\{\s*"instruction":\s*"(.*?)(?<!\\)",' \
              r'\s*"input":\s*"(.*?)(?<!\\)",' \
              r'\s*"output":\s*"(.*?)(?<!\\)"\s*\}'

    # Use re.DOTALL to make . match newlines as well
    matches = re.findall(pattern, text, re.DOTALL)

    # Process each match into a dictionary
    result = []
    split_count = 0

    for i, (instr, inp, output) in enumerate(matches):
        # Unescape escaped quotes and process any special characters
        instr = instr.replace('\\"', '"').replace('\\\\', '\\')
        inp = inp.replace('\\"', '"').replace('\\\\', '\\')
        output = output.replace('\\"', '"').replace('\\\\', '\\')

        # Calculate approximate token count
        system_msg = "You are a helpful assistant."
        user_msg = instr + (f"\n\n{inp}" if inp else "")
        total_tokens = count_tokens(system_msg) + count_tokens(user_msg) + count_tokens(output)

        # If the example is within the token limit, add it as is
        if total_tokens <= max_tokens:
            entry = {
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": output}
                ]
            }
            result.append(entry)
        else:
            # Example is too long, split it
            print(f"Example {i + 1} exceeds token limit ({total_tokens} tokens). Splitting into multiple examples.")

            # For inputs, keep the instruction the same but split the content if needed
            user_msg_tokens = count_tokens(user_msg)

            # If just the input is too long (very rare), we must split it
            if user_msg_tokens > max_tokens // 2:  # Allow some room for system and assistant messages
                msg_splits = split_text(user_msg, max_tokens // 2)

                # For simplicity, we'll use the first part with the full output
                # More sophisticated approaches would split both input and output coherently
                for j, msg_part in enumerate(msg_splits):
                    if j == 0:
                        # First part gets the full output
                        entry = {
                            "messages": [
                                {"role": "system", "content": system_msg},
                                {"role": "user", "content": f"{msg_part} (Part 1/{len(msg_splits)})"},
                                {"role": "assistant", "content": output}
                            ]
                        }
                    else:
                        # Other parts get a placeholder response
                        entry = {
                            "messages": [
                                {"role": "system", "content": system_msg},
                                {"role": "user", "content": f"{msg_part} (Part {j + 1}/{len(msg_splits)})"},
                                {"role": "assistant", "content": "This is a continuation from the previous part."}
                            ]
                        }
                    result.append(entry)
                split_count += 1
            else:
                # If the output is too long, split it into chunks
                output_splits = split_text(output, max_tokens - user_msg_tokens - count_tokens(system_msg))

                for j, output_part in enumerate(output_splits):
                    # Add a note to continuing parts
                    part_note = f" (Part {j + 1}/{len(output_splits)})" if len(output_splits) > 1 else ""

                    entry = {
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg + (f" (Continued from part {j})" if j > 0 else "")},
                            {"role": "assistant", "content": output_part + part_note}
                        ]
                    }
                    result.append(entry)
                split_count += 1

    if split_count > 0:
        print(f"Split {split_count} examples into multiple parts due to token limit constraints")

    return result


def split_data(data, validation_ratio=0.1):
    """Split data into training and validation sets."""
    # Shuffle the data to ensure randomness
    data_copy = data.copy()
    random.shuffle(data_copy)

    # Calculate the split point
    validation_size = max(1, int(len(data_copy) * validation_ratio))

    # Split the data
    validation_data = data_copy[:validation_size]
    training_data = data_copy[validation_size:]

    return training_data, validation_data


def write_jsonl(data, output_file):
    """Write the data to a JSONL file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def save_config(config, output_file):
    """Save the configuration to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Convert training data to JSONL format for ChatGPT-4o fine-tuning')
    parser.add_argument('input_file', type=str, help='Input file containing the training data')
    parser.add_argument('--output_train', type=str, default='training_data.jsonl',
                        help='Output JSONL file for training data')
    parser.add_argument('--output_val', type=str, default='validation_data.jsonl',
                        help='Output JSONL file for validation data')
    parser.add_argument('--config_file', type=str, default='fine_tune_config.json',
                        help='Output JSON file for fine-tuning configuration')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of data to use for validation (0-1)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for fine-tuning (typically 4-64)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate for fine-tuning (typically 1e-6 to 1e-4)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for fine-tuning')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--max_tokens', type=int, default=4096, help='Maximum token limit for examples')
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {args.input_file} does not exist")
        return

    # Read the input file
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Parse the data with token limit
    data = parse_training_data(text, max_tokens=args.max_tokens)
    print(f"Extracted {len(data)} valid examples")

    # Split into training and validation sets
    train_data, val_data = split_data(data, args.val_ratio)
    print(f"Split into {len(train_data)} training examples and {len(val_data)} validation examples")

    # Write to JSONL files
    write_jsonl(train_data, args.output_train)
    write_jsonl(val_data, args.output_val)
    print(f"Successfully wrote training data to {args.output_train}")
    print(f"Successfully wrote validation data to {args.output_val}")

    # Save fine-tuning configuration
    config = {
        "model": "gpt-4o",
        "training_file": args.output_train,
        "validation_file": args.output_val,
        "hyperparameters": {
            "batch_size": args.batch_size,
            "learning_rate_multiplier": args.learning_rate,
            "n_epochs": args.epochs
        },
        "validation_ratio": args.val_ratio,
        "random_seed": args.seed,
        "data_info": {
            "training_examples": len(train_data),
            "validation_examples": len(val_data),
            "total_examples": len(data)
        }
    }
    save_config(config, args.config_file)
    print(f"Saved fine-tuning configuration to {args.config_file}")


if __name__ == "__main__":
    main()