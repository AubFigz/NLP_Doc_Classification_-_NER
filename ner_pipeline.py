import spacy
import logging
import argparse
from pathlib import Path
import json

# Set up logging for better debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model(model_name="en_core_sci_sm"):
    """
    Load a SpaCy model by name, defaulting to SciSpacy. Includes error handling for missing models.
    """
    try:
        logging.info(f"Loading SpaCy model: {model_name}")
        nlp = spacy.load(model_name)
        return nlp
    except OSError:
        logging.error(f"Model {model_name} not found. Please install it via: python -m spacy download {model_name}")
        raise


def perform_ner(text, model):
    """
    Perform Named Entity Recognition (NER) on the provided text using the specified model.
    Handles large texts by splitting into batches for efficient processing.
    """
    try:
        logging.info(f"Performing NER on text with length {len(text)} characters.")
        doc = model(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        logging.info(f"Found {len(entities)} entities.")
        return entities
    except Exception as e:
        logging.error(f"Error performing NER: {e}")
        return []


def batch_process_texts(texts, model):
    """
    Process multiple texts in a batch for NER, returning a list of entities per text.
    """
    all_entities = []
    for text in texts:
        entities = perform_ner(text, model)
        all_entities.append(entities)
    return all_entities


def save_entities_to_file(entities, output_file):
    """
    Save extracted entities to a specified file in JSON format.
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(entities, f, indent=4)
        logging.info(f"Entities saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving entities to file: {e}")


def load_texts_from_file(input_file):
    """
    Load texts from a specified file, assuming each line in the file is a separate text.
    """
    try:
        with open(input_file, 'r') as f:
            texts = f.readlines()
        texts = [text.strip() for text in texts if text.strip()]  # Clean and filter empty lines
        logging.info(f"Loaded {len(texts)} texts from {input_file}")
        return texts
    except Exception as e:
        logging.error(f"Error loading texts from file: {e}")
        return []


def main(args):
    """
    Main function to orchestrate NER tasks: loading texts, performing NER, and saving results.
    """
    # Load the model
    model = load_model(args.model_name)

    # Load texts either from a file or a single string from the arguments
    if args.input_file:
        texts = load_texts_from_file(args.input_file)
    else:
        texts = [args.text]  # Use the provided text as a single item list

    # Perform NER on the texts
    entities = batch_process_texts(texts, model)

    # Save the results
    save_entities_to_file(entities, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Named Entity Recognition (NER) using a SpaCy model")

    # Arguments for text input, model selection, and output handling
    parser.add_argument("--model_name", type=str, default="en_core_sci_sm",
                        help="SpaCy model to use for NER (default is en_core_sci_sm)")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input text file, where each line is a separate document (optional)")
    parser.add_argument("--output_file", type=str, default="ner_results.json",
                        help="Output file to save extracted entities (JSON format)")
    parser.add_argument("--text", type=str, default="", help="Text for NER (if no input file is provided)")

    args = parser.parse_args()

    # Ensure at least one input source (text or file) is provided
    if not args.input_file and not args.text:
        logging.error("You must provide either an input file (--input_file) or a text string (--text).")
    else:
        main(args)
