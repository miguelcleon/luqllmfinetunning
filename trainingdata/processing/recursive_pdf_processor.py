
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
            lines = text.strip().split('\n')
            if lines and len(lines[0]) < 200:  # Reasonable title length
                metadata['title'] = lines[0].strip()

        # Try to extract abstract
        abstract = ""
        for page in doc[:2]:  # Check first two pages
            text = page.get_text()
            abstract_match = re.search(r'Abstract[:\.\s]+(.*?)(?:\n\n|\nIntroduction|\n1\.)', 
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
    section_pattern = re.compile(r'\n(\d+\.?\s+[A-Z][a-zA-Z\s]+|\b(?:Abstract|Introduction|Methods|Results|Discussion|Conclusion|References)\b)[:\.\s\n]+', re.IGNORECASE)

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
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)

    # Remove page numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    # Remove form feed characters
    text = re.sub(r'\f', '', text)

    # Fix hyphenated words at line breaks
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # Handle common PDF artifacts
    text = re.sub(r'\(cid:\d+\)', '', text)

    # Fix spacing after periods
    text = re.sub(r'\.([A-Z])', r'. \1', text)

    # Clean up citations
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    text = re.sub(r'\(\w+ et al\.,? \d{4}[a-z]?\)', '', text)

    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

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
            text += page.get_text() + "\n"
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
                        "text": f"<section>{section_name}</section>\n{section_text}"
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
