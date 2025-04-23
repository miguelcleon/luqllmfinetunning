import os
from PyPDF2 import PdfReader
from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)  # Use GPU


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text: {e}")
    return text


def summarize_text(text, max_length=150, min_length=30):
    """Summarize the given text."""
    # Check if text is empty or too short
    if not text or len(text) < 100:
        return "Text too short or empty to summarize."

    try:
        # Split text into chunks if it's too long
        max_tokens = 1024  # BART model limit
        if len(text) > max_tokens:
            text = text[:max_tokens]

        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Summarization error: {e}"


def process_papers(directory):
    """Process all PDF files in the given directory and its subdirectories."""
    for root, _, files in os.walk(directory):  # Fixed the underscore
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)  # Fixed asterisk to underscore
                print(f"Processing {pdf_path}")

                try:
                    # Extract text from PDF
                    text = extract_text_from_pdf(pdf_path)

                    if not text:
                        print(f"No text extracted from {pdf_path}")
                        continue

                    # Summarize the text
                    summary = summarize_text(text)

                    # Print or save the summary
                    print(f"Summary for {file}:\n{summary}\n")

                    # Optional: Save summary to file
                    summary_path = pdf_path.replace('.pdf', '_summary.txt')
                    with open(summary_path, 'w') as f:
                        f.write(summary)

                except Exception as e:
                    print(f"Failed to process {pdf_path}: {e}")


# Define the directory containing the research papers
papers_directory = '/mnt/c/one/OneDrive - USNH/LUQ-LTER-Share/publications'

# Process the papers
process_papers(papers_directory)