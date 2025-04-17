import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import argparse
import json
import os
import re
import logging
# python npu-enhanced-inference.py \
#  --model_path luq_lter_model_npu/final_model \
#  --base_model google/gemma-7b \
#  --publications_file combined_publications.json \
#  --use_npu \
#  --quantization int8 \
#  --max_length 512 \
#  --temperature 0.7 \
#  --interactive
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
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Gemma model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--base_model", type=str, default="google/gemma-7b", help="Base model name")
    parser.add_argument("--publications_file", type=str, help="Path to the combined publications JSON file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--use_npu", action="store_true", help="Use Intel NPU acceleration if available")
    parser.add_argument("--quantization", type=str, choices=["int8", "int4", "none"],
                        default="int8", help="Quantization type for NPU acceleration")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length for generated responses")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    return parser.parse_args()


def load_model_and_tokenizer(model_path, base_model, use_npu=False, quantization="int8"):
    """Load the fine-tuned model and tokenizer with optional NPU acceleration."""
    # Check NPU availability if requested
    has_npu, npu_lib = False, None
    if use_npu:
        has_npu, npu_lib = check_npu_availability()
        if has_npu:
            logger.info("NPU acceleration enabled for inference")
        else:
            logger.warning("NPU acceleration requested but not available. Falling back to CPU/GPU.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Set up quantization if using GPU
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False
        )
    else:
        bnb_config = None

    # Load base model and tokenizer
    logger.info(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto" if device == "cuda" else "cpu",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        quantization_config=bnb_config if device == "cuda" else None
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA weights
    logger.info(f"Loading LoRA weights from: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)

    # Apply NPU optimization if available
    if has_npu and npu_lib:
        try:
            logger.info(f"Optimizing model for NPU with quantization: {quantization}")

            # Prepare NPU optimization
            if quantization == "int8":
                logger.info("Using INT8 quantization for NPU")
                optimized_model = npu_lib.compile(model, dtype=torch.int8)
            elif quantization == "int4":
                logger.info("Using INT4 quantization for NPU")
                # Special handling for INT4 quantization
                from intel_npu_acceleration_library.compiler import CompilerConfig
                compiler_conf = CompilerConfig(dtype=torch.float16)  # Will be quantized to int4 internally
                optimized_model = npu_lib.compile(model, compiler_conf)
            else:
                logger.info("Using FP16 for NPU")
                optimized_model = npu_lib.compile(model)

            logger.info("Model successfully optimized for NPU")
            model = optimized_model
        except Exception as e:
            logger.warning(f"Failed to optimize for NPU: {e}")
            logger.info("Falling back to original model")

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
        logger.error(f"Error loading publications data: {e}")
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


def run_interactive_mode(pipe, tokenizer, publications_data, max_length, temperature):
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
                    section_text = sections[section_choice]
                    # Print first 1000 characters to avoid terminal flooding
                    print(section_text[:1000])
                    if len(section_text) > 1000:
                        print("\n... (truncated) ...")
                        more = input("\nShow more? (y/n): ")
                        if more.lower() == 'y':
                            print(section_text[1000:])
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
        response = generate_response(pipe, prompt, tokenizer, max_length, temperature)
        print("\nResponse:")
        print("-" * 50)
        print(response)
        print("-" * 50)


def main():
    # Parse arguments
    args = parse_args()

    # Load model and tokenizer with NPU if available
    pipe, tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.base_model,
        args.use_npu,
        args.quantization
    )

    # Load publications data if provided
    publications_data = None
    if args.publications_file:
        publications_data = load_publications_data(args.publications_file)
        if publications_data:
            logger.info(f"Loaded publications data with {len(publications_data['paper_index'])} papers")
        else:
            logger.warning("Failed to load publications data")

    if args.interactive:
        run_interactive_mode(pipe, tokenizer, publications_data, args.max_length, args.temperature)
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
            response = generate_response(pipe, prompt, tokenizer, args.max_length, args.temperature)
            print("\nResponse:")
            print("-" * 50)
            print(response)
            print("-" * 50)


if __name__ == "__main__":
    main()