import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
import logging
import argparse
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_wordcloud(text_data, save_path=None):
    """
    Plot a word cloud of the input text data and optionally save it to a file.
    :param text_data: Concatenated text data for word cloud generation
    :param save_path: Path to save the word cloud image (optional)
    """
    logging.info("Generating word cloud...")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')

    if save_path:
        plt.savefig(save_path)
        logging.info(f"Word cloud saved to {save_path}")
    else:
        plt.show()


def plot_top_terms(text_data, num_terms=10, save_path=None):
    """
    Plot the top N most common terms in the text data using a bar plot.
    :param text_data: Concatenated text data to extract top terms from
    :param num_terms: Number of top terms to plot (default is 10)
    :param save_path: Path to save the bar plot image (optional)
    """
    logging.info(f"Generating bar plot for top {num_terms} terms...")
    words = text_data.split()
    word_freq = Counter(words)
    common_words = word_freq.most_common(num_terms)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=[word[0] for word in common_words], y=[word[1] for word in common_words])
    plt.title(f"Top {num_terms} Common Words")

    if save_path:
        plt.savefig(save_path)
        logging.info(f"Top terms plot saved to {save_path}")
    else:
        plt.show()


def load_text_from_file(file_path):
    """
    Load text data from a specified file.
    :param file_path: Path to the text file
    :return: Loaded text data as a single concatenated string
    """
    if not os.path.isfile(file_path):
        logging.error(f"File not found: {file_path}")
        return ""

    logging.info(f"Loading text data from {file_path}...")
    with open(file_path, 'r') as file:
        text_data = file.read()
    return text_data


def main(args):
    # Load text data from the input file or from the text argument
    if args.input_file:
        text_data = load_text_from_file(args.input_file)
    else:
        text_data = args.text

    if not text_data:
        logging.error("No text data available to analyze.")
        return

    # Plot word cloud and save it to file if specified
    if args.wordcloud_output:
        plot_wordcloud(text_data, save_path=args.wordcloud_output)
    else:
        plot_wordcloud(text_data)

    # Plot top terms and save it to file if specified
    if args.top_terms_output:
        plot_top_terms(text_data, num_terms=args.num_top_terms, save_path=args.top_terms_output)
    else:
        plot_top_terms(text_data, num_terms=args.num_top_terms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Exploratory Data Analysis (EDA) on text data")

    # Add arguments for text input, file input, and output handling
    parser.add_argument("--input_file", type=str, help="Path to input text file")
    parser.add_argument("--text", type=str, default="", help="Text data for analysis (if no input file is provided)")
    parser.add_argument("--wordcloud_output", type=str, help="Path to save word cloud image (optional)")
    parser.add_argument("--top_terms_output", type=str, help="Path to save top terms bar plot (optional)")
    parser.add_argument("--num_top_terms", type=int, default=10, help="Number of top common terms to plot")

    args = parser.parse_args()

    # Ensure either a text string or an input file is provided
    if not args.input_file and not args.text:
        logging.error("You must provide either an input file (--input_file) or text data (--text).")
    else:
        main(args)
