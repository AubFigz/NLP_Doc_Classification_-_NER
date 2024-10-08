import os
import re
import hashlib
import logging
import spacy
import concurrent.futures
import nltk
from tqdm import tqdm
from pdfminer.high_level import extract_text
from nltk.corpus import stopwords
from pathlib import Path
import json
import threading

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download stopwords if not already available
nltk.download('stopwords')

# Load spacy model and stopwords globally (optionally on GPU if available)
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"]).to_gpu()  # Use GPU if available
except Exception:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Fall back to CPU

stop_words = set(stopwords.words('english'))


def hash_file(file_path):
    """
    Compute a hash for the given file to detect changes or avoid reprocessing the same content.
    """
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def pdf_to_text(pdf_file):
    """
    Extract text from a single PDF file.
    """
    try:
        logging.info(f"Extracting text from: {pdf_file}")
        text = extract_text(pdf_file)
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_file}: {e}")
        return ""


def clean_text(text):
    """
    Clean text by lowercasing, removing punctuation, lemmatizing, and removing stopwords.
    """
    try:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if token.is_alpha]
        tokens = [token for token in tokens if token not in stop_words]
        return " ".join(tokens)
    except Exception as e:
        logging.error(f"Error during text cleaning: {e}")
        return ""


def preprocess_paper(pdf_file):
    """
    Full pipeline for extracting and cleaning text from a single PDF file.
    Hash each file to avoid redundant processing if content hasn't changed.
    """
    try:
        pdf_hash = hash_file(pdf_file)
        cache_file = f'./cache/{pdf_hash}.json'

        if Path(cache_file).exists():
            logging.info(f"Using cached result for {pdf_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)

        # Extract and clean text
        raw_text = pdf_to_text(pdf_file)
        cleaned_text = clean_text(raw_text)

        # Save result to cache
        with open(cache_file, 'w') as f:
            json.dump(cleaned_text, f)

        return cleaned_text
    except Exception as e:
        logging.error(f"Error processing file {pdf_file}: {e}")
        return ""


def preprocess_papers(pdf_dir, max_workers=4):
    """
    Preprocess all PDF files in the specified directory using a combination of thread and process pools.
    """
    text_data = {}

    # Ensure cache directory exists
    Path('./cache').mkdir(parents=True, exist_ok=True)

    # Get a list of all PDF files in the directory
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

    if not pdf_files:
        logging.warning(f"No PDF files found in the directory: {pdf_dir}")
        return text_data

    # ThreadPoolExecutor to read files concurrently (I/O bound)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as thread_executor:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as process_executor:
            # Use threads for reading files and processes for cleaning them
            future_to_pdf = {process_executor.submit(preprocess_paper, pdf_file): pdf_file for pdf_file in pdf_files}

            for future in tqdm(concurrent.futures.as_completed(future_to_pdf), total=len(pdf_files)):
                pdf_file = future_to_pdf[future]
                try:
                    text_data[os.path.basename(pdf_file)] = future.result()
                except Exception as e:
                    logging.error(f"Error processing file {pdf_file}: {e}")

    logging.info(f"Processed {len(text_data)} PDFs.")
    return text_data


if __name__ == "__main__":
    # Define the PDF directory
    pdf_directory = './papers'

    # Preprocess all PDFs
    processed_data = preprocess_papers(pdf_directory, max_workers=8)

    # Display or save the processed data
    for file_name, text in processed_data.items():
        logging.info(
            f"Processed file: {file_name}\nText: {text[:500]}...\n")  # Display the first 500 characters of each text
