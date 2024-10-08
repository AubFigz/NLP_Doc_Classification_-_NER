Machine Learning Pipeline for Text Classification, Model Explainability, and AWS Deployment

Overview
This project is a comprehensive end-to-end machine learning pipeline that covers data preprocessing, text classification, model explainability, and deployment. It is designed to handle high-performance computing environments (HPC) with support for distributed training, mixed-precision, and large-scale NLP models. The project leverages BERT (Bidirectional Encoder Representations from Transformers) for text classification and SHAP (SHapley Additive exPlanations) for model explainability.

Key components of the project include:

Preprocessing of raw textual data, including PDF handling and tokenization.
Fine-tuning a BERT model for text classification using distributed and mixed-precision training.
Explainability of model predictions using SHAP.
Model deployment to AWS S3 for storage and later inference.
Exploratory Data Analysis (EDA) for understanding textual data patterns.
Named Entity Recognition (NER) for extracting key entities from scientific texts.

Requirements
To run this project, you need to install the required Python libraries. These can be installed using pip and the requirements.txt file.

Core Libraries:
PyTorch: For model training and deep learning tasks.
Transformers: Hugging Face's transformer library for BERT and other NLP models.
SHAP: For model explainability.
Spacy & SciSpacy: For Named Entity Recognition (NER).
AWS Boto3: To upload models to AWS S3.
Optuna: For hyperparameter tuning.
Matplotlib & Seaborn: For EDA visualizations.
For a full list of dependencies, check out the requirements.txt file in this repository.

Setup
Follow these steps to set up and run the project locally:

Clone the repository:
bash
git clone https://github.com/your-username/machine-learning-pipeline.git
cd machine-learning-pipeline

Install dependencies: Use pip to install the necessary Python libraries from the requirements.txt file:
bash
pip install -r requirements.txt

Set up AWS Credentials (Optional for AWS deployment): Ensure that your AWS credentials are configured in your environment. You can set them up via the AWS CLI:
bash
aws configure

Project Structure
.
├── data_preprocessing.py         # Preprocess raw data (e.g., from PDFs) and clean text
├── text_classification.py        # Fine-tune BERT for text classification
├── model_explainability.py       # Use SHAP to explain BERT's predictions
├── aws_deployment.py             # Upload the trained model to AWS S3
├── hpc_training.py               # Train BERT in a high-performance computing (HPC) environment
├── ner_pipeline.py               # Perform Named Entity Recognition (NER) using SciSpacy
├── eda.py                        # Generate EDA visualizations like word clouds and top terms
├── Dockerfile                    # Docker image for the project
├── requirements.txt              # Python dependencies for the project
└── README.md                     # Project overview and instructions

File Descriptions
1. data_preprocessing.py
This script is responsible for the preprocessing of raw text data, especially from PDF files. It:
Extracts text from PDFs.
Cleans the text (e.g., removes stopwords, punctuation).
Caches preprocessed text to avoid redundant work.
Uses parallel processing for faster performance on large datasets.

2. text_classification.py
This script fine-tunes a BERT model for text classification. Key features include:
Hyperparameter tuning using Optuna for optimal performance.
Support for early stopping and gradient accumulation for efficient training.
TensorBoard integration for real-time monitoring of training metrics.
Can run in both single-GPU and multi-GPU environments.

3. model_explainability.py
This script generates SHAP explanations for the fine-tuned BERT model's predictions. It:
Supports batch explanations for multiple input texts.
Visualizes SHAP values and saves the explanations as HTML files for easy sharing.
Handles both interactive mode (e.g., display SHAP plots) and batch mode (e.g., save to file).

4. aws_deployment.py
This script uploads the trained model to an AWS S3 bucket. Key features:
Progress tracking for large uploads using tqdm.
Error handling for file and credential issues.
Command-line interface for flexibility when specifying file paths and S3 bucket details.

5. hpc_training.py
This script is optimized for distributed and mixed-precision training on HPC environments. It:
Leverages torch.distributed for multi-GPU setups.
Includes checkpointing after each epoch for fault tolerance.
Supports mixed-precision training via torch.cuda.amp to speed up training while reducing memory usage.

6. ner_pipeline.py
This script performs Named Entity Recognition (NER) on scientific texts using SciSpacy. It:
Processes both individual text strings and batch files.
Extracts entities (e.g., chemicals, genes, diseases) from scientific text.
Saves results in JSON format for easy downstream processing.

7. eda.py
This script performs basic exploratory data analysis (EDA) on text data. It:
Generates visualizations like word clouds and bar plots of the most common terms.
Supports saving the generated visualizations as PNG files or displaying them interactively.
Loads text data either from a file or from a command-line input.

8. Dockerfile
The Dockerfile sets up a lightweight Python environment for running the project. It:
Installs only the necessary dependencies to reduce the image size.
Cleans up unnecessary files after installation to keep the container lightweight.
Allows for easy deployment in a Docker container, facilitating scalability and portability.

Running the Pipeline
1. Preprocess Text Data
Run the data_preprocessing.py script to extract and clean text from your raw data (e.g., PDFs):

bash
python data_preprocessing.py --pdf_dir "./data/pdfs"

2. Fine-Tune BERT for Text Classification
Train a BERT model on your text data with text_classification.py. Example:

bash
python text_classification.py --dataset_split train --batch_size 16 --epochs 4 --model_output_dir ./output_model

3. Model Explainability with SHAP
Explain BERT's predictions using SHAP:

bash
python model_explainability.py --model_dir "./output_model" --text "Your sample text here"

4. Deploy the Model to AWS S3
Upload the trained model to an S3 bucket for storage or inference:

bash
python aws_deployment.py --local_model_path "./output_model/pytorch_model.bin" --bucket_name "your-bucket" --s3_model_path "models/bert_model.bin"

5. Named Entity Recognition (NER)
Perform NER on a set of scientific texts using ner_pipeline.py:

bash
python ner_pipeline.py --input_file "data/scientific_texts.txt" --output_file "ner_results.json"

6. Exploratory Data Analysis (EDA)
Generate word clouds and term frequency plots for your text data:

bash
python eda.py --input_file "./data/sample_text.txt" --wordcloud_output "wordcloud.png" --top_terms_output "top_terms.png"

Docker Integration
To run the project in a Docker container, you can build the Docker image using the provided Dockerfile:

bash
docker build -t ml_pipeline_image .

Once the image is built, you can run the container:

bash
docker run -it ml_pipeline_image

This ensures that your project can run in a consistent environment, avoiding any dependency conflicts.
