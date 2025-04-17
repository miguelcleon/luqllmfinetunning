
# python PDFToJSON.py --input /path/to/papers --output papers_metadata.json --training_output training_data --format all

import os
import re
import json
import argparse
import hashlib
import string
import requests
from tqdm import tqdm
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import crossref_commons.retrieval
from typing import Dict, Any, List, Optional, Tuple

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def main():
    parser = argparse.ArgumentParser(description='Enhanced PDF processor for research papers with DOI metadata')
    parser.add_argument('--input', nargs='+', required=True,
                        help='One or more input directories containing PDFs (will be searched recursively)')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--min_length', type=int, default=500, help='Minimum text length to include (chars)')
    parser.add_argument('--metadata', action='store_true', help='Extract and include metadata', default=True)
    parser.add_argument('--sections', action='store_true', help='Segment papers into sections', default=True)
    parser.add_argument('--training_output', help='Base filename for training-ready JSON files')
    parser.add_argument('--format', choices=['basic', 'instruction', 'sections', 'relationships', 'all'],
                        default='sections', help='Training data format(s) to generate')

    args = parser.parse_args()

    # Check if all input paths are directories
    valid_dirs = []
    for input_path in args.input:
        if os.path.isdir(input_path):
            valid_dirs.append(input_path)
        else:
            print(f"Warning: {input_path} is not a valid directory. Skipping.")

    if not valid_dirs:
        print("Error: No valid directories provided")
        return

    # Process all directories and combine results
    documents = process_multiple_directories(
        valid_dirs,
        args.output,
        args.min_length,
        args.metadata,
        args.sections
    )

    # Create training data if requested
    if args.training_output and documents:
        if args.format == 'all':
            # Generate all formats with appropriate suffixes
            for format_type in ['basic', 'instruction', 'sections', 'relationships']:
                output_file = f"{args.training_output}_{format_type}.json"
                print(f"Generating {format_type} format training data...")
                create_training_json(args.output, output_file, format_type)
            print("All training formats generated successfully.")
        else:
            # Generate just the requested format
            create_training_json(args.output, args.training_output, args.format)


def extract_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a PDF file
    """
    metadata = {
        "source": pdf_path,
        "length": os.path.getsize(pdf_path),
        "metadata": {}
    }

    try:
        # Try to use PyMuPDF (fitz) for extraction
        doc = fitz.open(pdf_path)
        pdf_info = doc.metadata

        # Convert metadata to dictionary
        for key, value in pdf_info.items():
            if value and isinstance(value, str):
                metadata["metadata"][key.lower()] = value.strip()

        # Extract additional metadata from the PDF content
        first_page_text = doc[0].get_text()

        # Extract title from the first page if not in metadata or if it's generic
        if not metadata["metadata"].get('title') or metadata["metadata"].get('title') in ['Microsoft Word', 'Untitled',
                                                                                          '']:
            # Simple heuristic: look for the first substantial line
            lines = first_page_text.strip().split('\n')
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                if line and len(line) > 10 and len(line) < 200 and not re.match(r'^\d', line):
                    # Likely a title if it's reasonably sized and doesn't start with a number
                    metadata["metadata"]['title'] = line
                    break

        # Try to extract DOI
        doi_pattern = r'(?:DOI|doi)[\s:]*([0-9\.]+\/[a-zA-Z0-9\.\-_\/]+)'
        doi_match = re.search(doi_pattern, first_page_text)
        if doi_match:
            metadata["metadata"]['doi'] = doi_match.group(1).strip()

        # Try to extract publication year
        year_pattern = r'(?:©|\(c\)|\(C\)|Copyright|\b)[\s]*([12][0-9]{3})(?:\b|,)'
        year_match = re.search(year_pattern, first_page_text)
        if year_match:
            metadata["metadata"]['year'] = year_match.group(1).strip()

        # Extract authors
        authors = extract_authors(first_page_text)
        if authors:
            metadata["metadata"]['authors'] = authors
            metadata["metadata"]['author'] = ", ".join(authors)

        # Try to extract abstract
        abstract = ""
        for page in doc[:2]:  # Check first two pages
            text = page.get_text()
            abstract_match = re.search(
                r'(?:Abstract|ABSTRACT)[:\.\s]+(.*?)(?:\n\n|\n(?:Introduction|INTRODUCTION|Keywords|KEYWORDS|1\.))',
                text, re.DOTALL | re.IGNORECASE)
            if abstract_match:
                abstract = abstract_match.group(1).strip()
                break

        metadata["metadata"]['abstract'] = abstract

        doc.close()

        # Additional enhancement with DOI if needed
        metadata = enhance_metadata_with_doi(metadata)

        # Clean up the metadata
        metadata["metadata"] = clean_metadata(metadata["metadata"])

    except Exception as e:
        print(f"Error extracting metadata from {pdf_path}: {e}")

    return metadata


def extract_authors(text):
    """Extract author names from the paper."""
    # Try to find author list after title or before abstract
    author_patterns = [
        # Pattern for author list with affiliations
        r'(?:^|\n)(?!Abstract|Introduction|Keywords)([A-Z][a-zA-Z\-\s,\.]+(?:,|\sand\s|\s&\s)[A-Z][a-zA-Z\-\s,\.]+)(?=\n\s*\d|\n\s*\w+@|\n\s*Abstract|\n\s*\()',
        # Pattern for simple comma-separated author list
        r'(?<=\n)([A-Z][a-zA-Z\-\s]+(?:,\s*[A-Z][a-zA-Z\-\s]+)+)(?=\n)',
        # Pattern for authors with numbers for affiliations
        r'([A-Z][a-zA-Z\-\s]+(?:\s*\d+\s*,\s*[A-Z][a-zA-Z\-\s]+\s*\d+)+)',
    ]

    for pattern in author_patterns:
        author_match = re.search(pattern, text[:1000], re.MULTILINE)
        if author_match:
            authors_text = author_match.group(1).strip()
            # Clean up and split the author string
            authors_text = re.sub(r'\d+', '', authors_text)  # Remove numbers
            authors_text = re.sub(r'\([^)]*\)', '', authors_text)  # Remove parentheses

            # Split by common author separators
            if ',' in authors_text:
                authors = [a.strip() for a in authors_text.split(',')]
            elif ' and ' in authors_text:
                parts = authors_text.split(' and ')
                authors = []
                for part in parts:
                    if ',' in part:
                        authors.extend([a.strip() for a in part.split(',')])
                    else:
                        authors.append(part.strip())
            else:
                authors = [authors_text]

            # Clean authors list
            authors = [a for a in authors if a and len(a) > 1 and not a.isdigit()]
            return authors

    return []


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

    # Clean up headers/footers (often contains journal names, dates, page numbers)
    lines = text.split('\n')
    if len(lines) > 10:
        # Check for repeating headers/footers
        header_candidates = set(lines[:5])
        footer_candidates = set(lines[-5:])

        # Remove lines that appear in both header and body sections
        filtered_lines = []
        for i, line in enumerate(lines):
            if ((line in header_candidates and i > 5) or
                    (line in footer_candidates and i < len(lines) - 5) or
                    not any(line == x for x in lines if x != line)):
                filtered_lines.append(line)

        text = '\n'.join(filtered_lines)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean up metadata by removing unhelpful fields and standardizing values
    """
    # Remove unhelpful Adobe information
    unhelpful_patterns = [
        r'adobe',
        r'acrobat',
        r'distiller',
        r'indesign',
        r'itext',
        r'pdf library',
        r'pdf-\d+\.\d+',
        r'arbortext'
    ]

    fields_to_check = ['creator', 'producer']
    for field in fields_to_check:
        if field in metadata and metadata[field]:
            value = metadata[field]
            if any(re.search(pattern, value, re.IGNORECASE) for pattern in unhelpful_patterns):
                metadata[field] = ""

    # Fix empty author field
    if "author" in metadata and not metadata["author"]:
        if "authors" in metadata:
            # Use authors list if available
            metadata["author"] = ", ".join(metadata["authors"])

    return metadata


def fetch_doi_metadata(doi: str) -> Dict[str, Any]:
    """
    Fetch metadata from CrossRef or other DOI services
    """
    metadata = {}

    try:
        # Try using crossref-commons library first
        work = crossref_commons.retrieval.get_publication_as_json(doi)

        if 'author' in work:
            authors = []
            for author in work['author']:
                if 'given' in author and 'family' in author:
                    authors.append(f"{author['given']} {author['family']}")
                elif 'name' in author:
                    authors.append(author['name'])

            if authors:
                metadata['authors'] = authors

        if 'title' in work and work['title']:
            metadata['title'] = work['title'][0]

        if 'published-print' in work and 'date-parts' in work['published-print']:
            metadata['year'] = str(work['published-print']['date-parts'][0][0])

    except Exception as e:
        # Fallback to direct API request
        try:
            response = requests.get(f"https://api.crossref.org/works/{doi}")
            if response.status_code == 200:
                data = response.json()
                if 'message' in data:
                    msg = data['message']

                    if 'author' in msg:
                        authors = []
                        for author in msg['author']:
                            if 'given' in author and 'family' in author:
                                authors.append(f"{author['given']} {author['family']}")
                            elif 'name' in author:
                                authors.append(author['name'])

                        if authors:
                            metadata['authors'] = authors

                    if 'title' in msg and msg['title']:
                        metadata['title'] = msg['title'][0]

                    if 'published-print' in msg and 'date-parts' in msg['published-print']:
                        metadata['year'] = str(msg['published-print']['date-parts'][0][0])

        except Exception as fallback_err:
            print(f"Error fetching DOI metadata from fallback API: {fallback_err}")

    return metadata


def enhance_metadata_with_doi(pdf_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance PDF metadata with information from DOI when needed
    """
    metadata = pdf_metadata.copy()

    # Only fetch DOI metadata if authors are missing
    if "metadata" in metadata and "doi" in metadata["metadata"] and metadata["metadata"]["doi"]:
        doi = metadata["metadata"]["doi"]

        # Check if we need to fetch additional data
        needs_authors = "authors" not in metadata["metadata"] or not metadata["metadata"]["authors"]
        needs_author = "author" not in metadata["metadata"] or not metadata["metadata"]["author"]

        if needs_authors or needs_author:
            doi_metadata = fetch_doi_metadata(doi)

            if "authors" in doi_metadata and doi_metadata["authors"]:
                metadata["metadata"]["authors"] = doi_metadata["authors"]
                if needs_author:
                    metadata["metadata"]["author"] = ", ".join(doi_metadata["authors"])

            if "title" in doi_metadata and doi_metadata["title"] and (
                    "title" not in metadata["metadata"] or not metadata["metadata"]["title"]):
                metadata["metadata"]["title"] = doi_metadata["title"]

            if "year" in doi_metadata and doi_metadata["year"] and (
                    "year" not in metadata["metadata"] or not metadata["metadata"]["year"]):
                metadata["metadata"]["year"] = doi_metadata["year"]

    return metadata


def segment_paper(text):
    """Segment paper into sections with improved detection."""
    sections = {}

    # Define common section headers with variations
    section_headers = [
        'Abstract', 'ABSTRACT',
        'Introduction', 'INTRODUCTION',
        'Background', 'BACKGROUND',
        'Literature Review', 'LITERATURE REVIEW',
        'Related Work', 'RELATED WORK',
        'Materials and Methods', 'MATERIALS AND METHODS', 'Methods', 'METHODS', 'Methodology', 'METHODOLOGY',
        'Experimental Setup', 'EXPERIMENTAL SETUP', 'Experiments', 'EXPERIMENTS',
        'Results', 'RESULTS',
        'Discussion', 'DISCUSSION',
        'Results and Discussion', 'RESULTS AND DISCUSSION',
        'Conclusion', 'CONCLUSION', 'Conclusions', 'CONCLUSIONS',
        'Acknowledgment', 'ACKNOWLEDGMENT', 'Acknowledgments', 'ACKNOWLEDGMENTS', 'Acknowledgements',
        'ACKNOWLEDGEMENTS',
        'References', 'REFERENCES', 'Bibliography', 'BIBLIOGRAPHY',
        'Appendix', 'APPENDIX', 'Appendices', 'APPENDICES'
    ]

    # Create pattern for numbered sections and standard headers
    numbered_section_pattern = r'\n(?:(\d+\.(?:\d+)*)\s+([A-Z][a-zA-Z\s]+)|({}))(?:[:\.\s\n]+)'.format(
        '|'.join(section_headers))

    # Find all section headers
    matches = list(re.finditer(numbered_section_pattern, text, re.MULTILINE))

    if not matches:
        # If no sections found, return the whole text as "body"
        return {"body": text}

    # Process each section
    for i, match in enumerate(matches):
        section_number = match.group(1) if match.group(1) else ""
        section_name = match.group(2) if match.group(2) else match.group(3)

        # Full section name with number if applicable
        full_section_name = f"{section_number} {section_name}" if section_number else section_name

        start_pos = match.end()

        # Get end position (start of next section or end of text)
        if i < len(matches) - 1:
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)

        section_text = text[start_pos:end_pos].strip()
        sections[full_section_name] = section_text

    return sections


def normalize_author_name(name):
    """Normalize author names for consistent matching."""
    # Convert to lowercase and remove punctuation
    name = name.lower()
    name = name.translate(str.maketrans('', '', string.punctuation))

    # Handle common name formats
    parts = name.split()
    if len(parts) >= 2:
        # Try to handle lastname, firstname format
        if ',' in name:
            lastname_first_parts = name.split(',')
            if len(lastname_first_parts) >= 2:
                lastname = lastname_first_parts[0].strip()
                firstname = lastname_first_parts[1].strip()
                # Use only first initial if available
                if firstname:
                    firstname_initial = firstname[0]
                    return f"{lastname} {firstname_initial}"

        # Handle firstname lastname format - get last part as surname, first initial
        lastname = parts[-1]
        firstname_initial = parts[0][0]
        return f"{lastname} {firstname_initial}"

    return name


def build_author_relationships(documents):
    """Build a graph of author relationships from documents."""
    author_papers = defaultdict(list)
    paper_authors = {}

    # Build author-paper relationships
    for doc in documents:
        if "metadata" in doc and "authors" in doc["metadata"]:
            authors = doc["metadata"]["authors"]
            normalized_authors = [normalize_author_name(author) for author in authors]

            # Store the mapping
            paper_id = doc["id"]
            paper_authors[paper_id] = normalized_authors

            # Add paper to each author's list
            for author in normalized_authors:
                author_papers[author].append(paper_id)

    # Build co-authorship graph
    coauthor_graph = defaultdict(set)

    for paper_id, authors in paper_authors.items():
        for author in authors:
            # Each author is connected to all other authors of the same paper
            for coauthor in authors:
                if author != coauthor:
                    coauthor_graph[author].add(coauthor)

    return {
        "author_papers": dict(author_papers),
        "paper_authors": paper_authors,
        "coauthor_graph": {author: list(coauthors) for author, coauthors in coauthor_graph.items()}
    }


def process_pdf(pdf_path, include_metadata=True, include_sections=True):
    """Process a single PDF with advanced options."""
    try:
        # Basic text extraction with PyMuPDF
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        doc.close()

        # Clean the text
        clean_full_text = clean_text(full_text)

        # Extract metadata if requested
        metadata = {}
        if include_metadata:
            metadata = extract_pdf_metadata(pdf_path)["metadata"]

        # Segment into sections if requested
        sections = {}
        if include_sections:
            sections = segment_paper(clean_full_text)

        # Create unique ID for document based on content
        content_hash = hashlib.md5(clean_full_text.encode()).hexdigest()[:12]

        # Create document entry
        document = {
            "id": content_hash,
            "text": clean_full_text,
            "source": pdf_path,
            "length": len(clean_full_text)
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
    """Process all PDFs in a directory and save as a JSON file with author relationships."""
    # Find all PDF files recursively
    pdf_files = find_pdf_files(dir_path)

    if not pdf_files:
        print(f"No PDF files found in {dir_path} and its subdirectories.")
        return []

    # Process each PDF
    print(f"Processing {len(pdf_files)} PDF files...")
    documents = []

    for pdf_file in tqdm(pdf_files):
        document = process_pdf(pdf_file, include_metadata, include_sections)

        if document and document.get("text") and len(document["text"]) >= min_length:
            documents.append(document)
        else:
            print(f"Skipping {pdf_file}: Text too short or extraction failed")

    if not documents:
        print("No valid documents were processed.")
        return []

    # Build author relationships
    author_relationships = build_author_relationships(documents)

    # Combine documents and relationships
    dataset = {
        "documents": documents,
        "author_relationships": author_relationships
    }

    # Save as JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(documents)} PDFs successfully out of {len(pdf_files)} total")
    print(f"Output saved to {output_file}")

    return documents


def process_multiple_directories(input_dirs, output_file, min_length=500, include_metadata=True, include_sections=True):
    """Process PDFs from multiple directories and combine into a single dataset."""
    all_documents = []

    # Process each directory separately
    for dir_path in input_dirs:
        print(f"\nProcessing directory: {dir_path}")

        # Check if directory exists
        if not os.path.isdir(dir_path):
            print(f"Warning: Directory {dir_path} does not exist or is not accessible. Skipping.")
            continue

        # Find all PDF files in this directory
        pdf_files = find_pdf_files(dir_path)

        if not pdf_files:
            print(f"No PDF files found in {dir_path} and its subdirectories.")
            continue

        # Process each PDF
        print(f"Processing {len(pdf_files)} PDF files...")

        for pdf_file in tqdm(pdf_files):
            document = process_pdf(pdf_file, include_metadata, include_sections)

            if document and document.get("text") and len(document["text"]) >= min_length:
                all_documents.append(document)
            else:
                print(f"Skipping {pdf_file}: Text too short or extraction failed")

    if not all_documents:
        print("No valid documents were processed from any directory.")
        return []

    # Build author relationships across all documents
    author_relationships = build_author_relationships(all_documents)

    # Combine documents and relationships
    dataset = {
        "documents": all_documents,
        "author_relationships": author_relationships
    }

    # Save as JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(all_documents)} PDFs successfully across all directories")
    print(f"Output saved to {output_file}")

    return all_documents


def create_training_json(input_json, output_json, format_type="sections"):
    """
    Convert processed JSON to training format with enhanced section awareness.

    format_type options:
    - "basic": Simple text entries
    - "instruction": Q&A format for instruction tuning
    - "sections": Section-based format with enhanced structure
    - "relationships": Include author relationship information in prompts
    """
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = data.get("documents", [])
        if not documents and isinstance(data, list):
            # Handle legacy format where it's just a list of documents
            documents = data

        author_relationships = data.get("author_relationships", {})

        if not documents:
            print(f"Warning: Input JSON file {input_json} is empty or has no documents.")
            return []
    except Exception as e:
        print(f"Error reading input JSON {input_json}: {e}")
        return []

    training_data = []

    for doc in documents:
        doc_id = doc.get("id", "")
        metadata = doc.get("metadata", {})
        title = metadata.get("title", "")
        authors = metadata.get("authors", [])
        abstract = metadata.get("abstract", "")
        sections = doc.get("sections", {})
        full_text = doc.get("text", "")

        if format_type == "basic":
            # Simple format with just text
            training_data.append({"text": full_text})

        elif format_type == "instruction":
            # Instruction tuning format with enhanced Q&A pairs

            # First, include the main document summary task
            if title and abstract:
                training_data.append({
                    "instruction": f"Summarize the paper titled '{title}' with the abstract: {abstract}",
                    "input": "",
                    "output": full_text
                })
            elif title:
                training_data.append({
                    "instruction": f"Summarize the paper titled '{title}'",
                    "input": "",
                    "output": full_text
                })
            elif abstract:
                training_data.append({
                    "instruction": f"Explain the following research abstract in detail: {abstract}",
                    "input": "",
                    "output": full_text
                })
            else:
                training_data.append({
                    "instruction": "Summarize this research paper",
                    "input": "",
                    "output": full_text
                })

            # Add section-specific Q&A pairs
            if sections:
                for section_name, section_text in sections.items():
                    if section_name.lower() not in ["abstract", "references", "bibliography", "acknowledgments",
                                                    "appendix"]:
                        training_data.append({
                            "instruction": f"In the paper {title if title else 'about this research'}, what does the section '{section_name}' discuss?",
                            "input": abstract if abstract else "",
                            "output": section_text
                        })

                        # Add some specific question types based on section
                        if "method" in section_name.lower() or "methodology" in section_name.lower():
                            training_data.append({
                                "instruction": f"What methods were used in the research '{title if title else 'described in this paper'}'?",
                                "input": abstract if abstract else "",
                                "output": section_text
                            })

                        if "result" in section_name.lower():
                            training_data.append({
                                "instruction": f"What were the main results of the study '{title if title else 'in this paper'}'?",
                                "input": abstract if abstract else "",
                                "output": section_text
                            })

                        if "discussion" in section_name.lower() or "conclusion" in section_name.lower():
                            training_data.append({
                                "instruction": f"What are the main conclusions of the paper '{title if title else 'on this research'}'?",
                                "input": abstract if abstract else "",
                                "output": section_text
                            })

        elif format_type == "sections":
            # Enhanced format with section markers for better structure awareness

            # Create a structured document with metadata and sections
            document_parts = []

            # Add metadata header
            if title:
                document_parts.append(f"<title>{title}</title>")

            if authors:
                document_parts.append(f"<authors>{', '.join(authors)}</authors>")

            if abstract:
                document_parts.append(f"<abstract>{abstract}</abstract>")

            # Add each section with proper tags
            if sections:
                for section_name, section_text in sections.items():
                    # Skip empty sections
                    if not section_text or len(section_text.strip()) < 50:
                        continue

                    # Clean section name for tag
                    clean_section_name = section_name.lower().replace(' ', '_')
                    document_parts.append(f"<section name=\"{section_name}\">{section_text}</section>")
            else:
                # If no sections available, add the full text
                document_parts.append(f"<body>{full_text}</body>")

            # Combine all parts into one structured document
            structured_document = "\n\n".join(document_parts)
            training_data.append({"text": structured_document})

        elif format_type == "relationships":
            # Format that includes author relationship information

            # Get coauthors for the authors of this paper
            paper_author_info = []
            if authors and doc_id in author_relationships.get("paper_authors", {}):
                norm_authors = author_relationships["paper_authors"][doc_id]

                # For each author, find their other papers and coauthors
                for i, author in enumerate(authors):
                    if i >= len(norm_authors):
                        continue

                    norm_author = norm_authors[i]

                    # Get other papers by this author
                    other_papers = []
                    if norm_author in author_relationships.get("author_papers", {}):
                        paper_ids = author_relationships["author_papers"][norm_author]
                        # Find titles of other papers
                        for paper_id in paper_ids:
                            if paper_id != doc_id:  # Skip current paper
                                for other_doc in documents:
                                    if other_doc.get("id") == paper_id and "metadata" in other_doc:
                                        other_title = other_doc["metadata"].get("title", "")
                                        if other_title:
                                            other_papers.append(other_title)

                    # Get coauthors
                    coauthors = []
                    if norm_author in author_relationships.get("coauthor_graph", {}):
                        norm_coauthors = author_relationships["coauthor_graph"][norm_author]
                        # Map back to original author names where possible
                        for norm_coauthor in norm_coauthors:
                            for other_author in authors:
                                if normalize_author_name(
                                        other_author) == norm_coauthor and other_author not in coauthors:
                                    coauthors.append(other_author)

                    # Add author info
                    author_info = {
                        "name": author,
                        "other_papers": other_papers[:5],  # Limit to 5 papers
                        "coauthors": coauthors
                    }
                    paper_author_info.append(author_info)

            # Create training examples with relationship info
            if title and abstract and paper_author_info:
                # Create a prompt that includes author relationship info
                author_context = ""
                for author_info in paper_author_info:
                    author_context += f"Author: {author_info['name']}\n"
                    if author_info['other_papers']:
                        author_context += f"Other papers: {', '.join(author_info['other_papers'])}\n"
                    if author_info['coauthors']:
                        author_context += f"Frequent collaborators: {', '.join(author_info['coauthors'])}\n"
                    author_context += "\n"

                training_data.append({
                    "instruction": f"Given the following information about the authors and their work, summarize the paper titled '{title}' with the abstract: {abstract}",
                    "input": author_context,
                    "output": full_text
                })

                # Add additional examples focused on author relationships
                if len(paper_author_info) > 1:  # If multiple authors
                    authors_str = ", ".join([a["name"] for a in paper_author_info])
                    training_data.append({
                        "instruction": f"Describe the collaboration patterns of authors {authors_str} based on their publication history.",
                        "input": author_context,
                        "output": f"The authors {authors_str} have collaborated on the paper '{title}'. " +
                                  "".join([
                                              f"{author_info['name']} has also published papers such as {', '.join(author_info['other_papers'][:3])}. "
                                              if author_info['other_papers'] else "" for author_info in
                                              paper_author_info]) +
                                  "Their research focuses on topics related to " +
                                  (abstract[:100] + "..." if len(abstract) > 100 else abstract)
                    })
            else:
                # Fallback if no relationship data
                training_data.append({
                    "instruction": f"Summarize the paper titled '{title}'",
                    "input": abstract if abstract else "",
                    "output": full_text
                })

    # Save training data
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    print(f"Created training data with {len(training_data)} entries in {format_type} format")
    print(f"Output saved to {output_json}")

    return training_data

def main():
    parser = argparse.ArgumentParser(description='Enhanced PDF processor for research papers with DOI metadata')
    parser.add_argument('--input', nargs='+', required=True,
                        help='One or more input directories containing PDFs (will be searched recursively)')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--min_length', type=int, default=500, help='Minimum text length to include (chars)')
    parser.add_argument('--metadata', action='store_true', help='Extract and include metadata', default=True)
    parser.add_argument('--sections', action='store_true', help='Segment papers into sections', default=True)
    parser.add_argument('--training_output', help='Base filename for training-ready JSON files')
    parser.add_argument('--format', choices=['basic', 'instruction', 'sections', 'relationships', 'all'],
                        default='sections', help='Training data format(s) to generate')

    args = parser.parse_args()

    # Check if all input paths are directories
    valid_dirs = []
    for input_path in args.input:
        if os.path.isdir(input_path):
            valid_dirs.append(input_path)
        else:
            print(f"Warning: {input_path} is not a valid directory. Skipping.")

    if not valid_dirs:
        print("Error: No valid directories provided")

    # Process all directories and combine results
    documents = process_multiple_directories(
        valid_dirs,
        args.output,
        args.min_length,
        args.metadata,
        args.sections
    )

    # Create training data if requested
    if args.training_output and documents:
        if args.format == 'all':
            # Generate all formats with appropriate suffixes
            for format_type in ['basic', 'instruction', 'sections', 'relationships']:
                output_file = f"{args.training_output}_{format_type}.json"
                print(f"Generating {format_type} format training data...")
                create_training_json(args.output, output_file, format_type)
            print("All training formats generated successfully.")
        else:
            # Generate just the requested format
            create_training_json(args.output, args.training_output, args.format)


if __name__ == "__main__":
    main()