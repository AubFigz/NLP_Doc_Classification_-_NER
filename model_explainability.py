import shap
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging
import argparse
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model(model_dir, tokenizer_name="bert-base-uncased"):
    """
    Load the pre-trained model and tokenizer from a specified directory.
    :param model_dir: Path to the directory containing the trained model
    :param tokenizer_name: Name of the tokenizer to use (default: bert-base-uncased)
    """
    logging.info(f"Loading model from {model_dir} and tokenizer {tokenizer_name}.")
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer


def explain_text(texts, model, tokenizer, save_path=None):
    """
    Generate SHAP explanations for the given list of texts and visualize or save the results.
    :param texts: List of text strings to explain
    :param model: Pretrained Hugging Face model
    :param tokenizer: Tokenizer for text encoding
    :param save_path: Path to save the SHAP visualizations (optional)
    """
    logging.info("Generating SHAP explanations...")

    # Tokenize the input texts
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    # Initialize SHAP explainer
    explainer = shap.Explainer(model, tokenizer)

    # Generate SHAP values
    shap_values = explainer(inputs['input_ids'])

    # Visualize or save SHAP results
    if save_path:
        shap_html = shap.plots.text(shap_values, display=False)
        with open(save_path, "w") as f:
            f.write(shap_html)
        logging.info(f"SHAP explanations saved to {save_path}")
    else:
        shap.plots.text(shap_values)


def load_texts_from_file(file_path):
    """
    Load text data from a file, where each line represents a separate text.
    :param file_path: Path to the text file
    :return: List of text strings
    """
    if not os.path.isfile(file_path):
        logging.error(f"File not found: {file_path}")
        return []

    logging.info(f"Loading texts from {file_path}...")
    with open(file_path, 'r') as file:
        texts = [line.strip() for line in file if line.strip()]

    logging.info(f"Loaded {len(texts)} texts from {file_path}.")
    return texts


def main(args):
    # Load the trained model and tokenizer
    model, tokenizer = load_model(args.model_dir, args.tokenizer_name)

    # Load texts either from file or a provided list
    if args.input_file:
        texts = load_texts_from_file(args.input_file)
    else:
        texts = [args.text]  # Use a single text if no file is provided

    if not texts:
        logging.error("No text data provided for explanation.")
        return

    # Generate SHAP explanations and save if specified
    explain_text(texts, model, tokenizer, save_path=args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain model predictions using SHAP")

    # Add arguments for model loading, input texts, and output paths
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the directory containing the trained model")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased",
                        help="Tokenizer name to use (default: bert-base-uncased)")
    parser.add_argument("--input_file", type=str, help="Path to input file containing texts to explain")
    parser.add_argument("--text", type=str, default="",
                        help="Single text input for explanation (used if no input file is provided)")
    parser.add_argument("--output_file", type=str, help="Path to save SHAP explanations (optional)")

    args = parser.parse_args()

    # Ensure at least one input source (text or file) is provided
    if not args.input_file and not args.text:
        logging.error("You must provide either an input file (--input_file) or a single text string (--text).")
    else:
        main(args)
