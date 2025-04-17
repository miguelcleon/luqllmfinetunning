import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import argparse
import json
import os
import re
# python inference-script.py \
#  --model_path luq_lter_model/final_model \
#  --base_model google/gemma-7b \
#  --publications_file combined_publications.json \
#  --interactive

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Gemma model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--base_model", type=str, default="google/gemma-7b", help="Base model name")
    parser.add_argument("--publications_file", type=str, help="Path to the combined publications JSON file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    return parser.parse_args()


def load_model_and_tokenizer(model_path, base_model):
    """Load the fine-tuned model and tokenizer."""
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Set up quantization if using GPU
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False
        )
    else:
        bnb_config = None

    # Load base model and tokenizer
    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        quantization_config=bnb_config if torch.cuda.is_available() else None
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA weights
    print(f"Loading LoRA weights from: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)

    # Create a pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        max_new_tokens=512
    )

    return pipe, tokenizer


def load_publications_data(publications_file):
    """Load and process the publications data for reference."""
    if not publications_file or not os.path.exists(publications_file):
        return None

    try:
        with open(publications_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Build searchable index of papers and authors
        paper_index = {}
        author_index = {}

        for doc in data.get("documents", []):
            # Add to paper index
            doc_id = doc.get("id", "")
            title = doc.get("metadata", {}).get("title", "")
            if title and doc_id:
                paper_index[title.lower()] = doc

            # Add to author index
            authors = doc.get("metadata", {}).get("authors", [])
            for author in authors:
                if author not in author_index:
                    author_index[author.lower()] = []
                author_index[author.lower()].append(doc)

        return {
            "paper_index": paper_index,
            "author_index": author_index,
            "author_relationships": data.get("author_relationships", {})
        }
    except Exception as e:
        print(f"Error loading publications data: {e}")
        return None


def find_papers_by_author(author_name, publications_data):
    """Find papers by a specific author."""
    if not publications_data:
        return []

    author_name = author_name.lower()
    matching_papers = []

    # Try exact match first
    if author_name in publications_data["author_index"]:
        matching_papers = publications_data["author_index"][author_name]
    else:
        # Try partial match
        for author in publications_data["author_index"]:
            if author_name in author:
                matching_papers.extend(publications_data["author_index"][author])

    # Return just the titles and abstracts
    return [
        {
            "title": paper.get("metadata", {}).get("title", ""),
            "abstract": paper.get("metadata", {}).get("abstract", ""),
            "year": paper.get("metadata", {}).get("year", "")
        }
        for paper in matching_papers
    ]


def find_paper_by_title(title, publications_data):
    """Find a specific paper by title."""
    if not publications_data:
        return None

    title = title.lower()

    # Try exact match first
    if title in publications_data["paper_index"]:
        paper = publications_data["paper_index"][title]
        return paper

    # Try partial match
    for paper_title in publications_data["paper_index"]:
        if title in paper_title:
            return publications_data["paper_index"][paper_title]

    return None


def find_related_authors(author_name, publications_data):
    """Find authors who frequently collaborate with the given author."""
    if not publications_data or "author_relationships" not in publications_data:
        return []

    author_name = author_name.lower()
    coauthor_graph = publications_data["author_relationships"].get("coauthor_graph", {})

    # Find matching author name
    matched_author = None
    for norm_author in coauthor_graph:
        if author_name in norm_author:
            matched_author = norm_author
            break

    if matched_author and matched_author in coauthor_graph:
        # Get coauthors
        return coauthor_graph[matched_author]

    return []


def generate_response(pipe, prompt, tokenizer, max_length=512, temperature=0.7, top_p=0.9):
    """Generate a response from the model."""
    # Format input in the instruction format for Gemma
    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    # Generate response
    outputs = pipe(
        formatted_prompt,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = outputs[0]['generated_text']

    # Extract just the model's response
    response_parts = generated_text.split("<start_of_turn>model\n")
    if len(response_parts) > 1:
        model_response = response_parts[1].split("<end_of_turn>")[0] if "<end_of_turn>" in response_parts[1] else \
        response_parts[1]
        return model_response
    else:
        return generated_text


def run_interactive_mode(pipe, tokenizer, publications_data):
    """Run interactive mode for the model."""
    print("\nFine-tuned Research Assistant - Interactive Mode")
    print("Enter 'q' or 'quit' to exit")
    print("Enter 'author:<name>' to find papers by an author")
    print("Enter 'paper:<title>' to find details about a specific paper")
    print("Enter 'related:<author>' to find related authors")
    print("-" * 50)

    while True:
        prompt = input("\nEnter a prompt: ")

        if prompt.lower() in ['q', 'quit', 'exit']:
            break

        # Check for special commands
        if prompt.startswith("author:"):
            author_name = prompt[7:].strip()
            papers = find_papers_by_author(author_name, publications_data)
            if papers:
                print(f"\nFound {len(papers)} papers by {author_name}:")
                for i, paper in enumerate(papers):
                    print(f"{i + 1}. {paper['title']} ({paper['year']})")
                    if paper['abstract']:
                        print(f"   Abstract: {paper['abstract'][:150]}...")
                    print()
            else:
                print(f"No papers found for author: {author_name}")
            continue

        elif prompt.startswith("paper:"):
            title = prompt[6:].strip()
            paper = find_paper_by_title(title, publications_data)
            if paper:
                print(f"\nPaper: {paper.get('metadata', {}).get('title', '')}")
                print(f"Authors: {', '.join(paper.get('metadata', {}).get('authors', []))}")
                print(f"Year: {paper.get('metadata', {}).get('year', '')}")
                print(f"Abstract: {paper.get('metadata', {}).get('abstract', '')}")

                # List sections
                sections = paper.get("sections", {})
                if sections:
                    print("\nSections:")
                    for section_name in sections:
                        print(f"- {section_name}")

                # Ask if user wants to see specific section
                section_choice = input("\nEnter section name to view or press Enter to continue: ")
                if section_choice and section_choice in sections:
                    print(f"\n{section_choice}:")
                    print(sections[section_choice])
            else:
                print(f"No paper found with title: {title}")
            continue

        elif prompt.startswith("related:"):
            author_name = prompt[8:].strip()
            related_authors = find_related_authors(author_name, publications_data)
            if related_authors:
                print(f"\nAuthors who collaborate with {author_name}:")
                for i, coauthor in enumerate(related_authors):
                    print(f"{i + 1}. {coauthor}")
            else:
                print(f"No collaborators found for author: {author_name}")
            continue

        # Regular prompt processing
        print("\nGenerating response...")
        response = generate_response(pipe, prompt, tokenizer)
        print("\nResponse:")
        print("-" * 50)
        print(response)
        print("-" * 50)


def main():
    args = parse_args()

    # Load model and tokenizer
    pipe, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)

    # Load publications data if provided
    publications_data = None
    if args.publications_file:
        publications_data = load_publications_data(args.publications_file)
        print(f"Loaded publications data with {len(publications_data['paper_index'])} papers")

    if args.interactive:
        run_interactive_mode(pipe, tokenizer, publications_data)
    else:
        # Simple test of the model
        test_prompts = [
            "What are the major research themes in tropical forest ecology?",
            "Summarize the key findings about biodiversity in tropical forests.",
            "What methods are used to measure climate change effects on ecosystems?",
            "Explain the relationship between soil nutrients and plant growth in tropical environments."
        ]

        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            response = generate_response(pipe, prompt, tokenizer)
            print("\nResponse:")
            print("-" * 50)
            print(response)
            print("-" * 50)


if __name__ == "__main__":
    main()