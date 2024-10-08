import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset, load_metric, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import logging
import argparse
import optuna
import numpy as np
from transformers.integrations import TensorBoardCallback
from transformers import EarlyStoppingCallback
from torch.utils.tensorboard import SummaryWriter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load metrics
accuracy_metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    """
    Compute metrics for model evaluation, such as accuracy, precision, recall, and F1 score.
    """
    logits, labels = eval_pred
    predictions = torch.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main(args):
    # Load dataset (can be replaced with a custom dataset)
    dataset = load_dataset("scientific_papers", split=args.dataset_split)

    # Train-test split
    datasets = dataset.train_test_split(test_size=0.1)
    train_val_datasets = datasets['train'].train_test_split(test_size=0.2)
    dataset_dict = DatasetDict({
        'train': train_val_datasets['train'],
        'validation': train_val_datasets['test'],
        'test': datasets['test']
    })

    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)

    # Preprocess the dataset
    def tokenize_function(example):
        return tokenizer(example['text'], padding="max_length", truncation=True)

    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True, num_proc=args.num_workers)

    # Use a data collator to dynamically pad the sequences
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Tensorboard setup
    writer = SummaryWriter(args.logging_dir)

    # Training arguments with custom settings
    training_args = TrainingArguments(
        output_dir=args.output_dir,  # Save checkpoints and models
        evaluation_strategy="epoch",
        save_strategy="epoch",  # Save checkpoints at the end of each epoch
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,  # Directory for logging
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,  # Limit the number of saved checkpoints
        load_best_model_at_end=True,  # Load the best model based on evaluation metrics
        fp16=args.fp16,  # Enable mixed precision training for speedup (if applicable)
        report_to="tensorboard",  # Reporting to TensorBoard for real-time metrics
        push_to_hub=False,  # Disable auto-push to Hugging Face hub
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Grad accumulation
        metric_for_best_model="accuracy",  # Use accuracy as the key metric
        eval_steps=args.eval_steps,
        logging_first_step=True,
        dataloader_num_workers=args.num_workers
    )

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), TensorBoardCallback(writer)]
        # Early stopping and TensorBoard integration
    )

    # Train the model
    logging.info("Starting training...")
    trainer.train()

    # Evaluate on test set
    logging.info("Evaluating on the test set...")
    metrics = trainer.evaluate(tokenized_datasets['test'])
    logging.info(f"Test set evaluation metrics: {metrics}")

    # Save the final model
    model.save_pretrained(args.model_output_dir)
    tokenizer.save_pretrained(args.model_output_dir)

    logging.info(f"Model saved to {args.model_output_dir}")


def objective(trial):
    """
    Optuna objective function to optimize hyperparameters.
    """
    args.epochs = trial.suggest_int("epochs", 2, 6)
    args.batch_size = trial.suggest_int("batch_size", 8, 32)
    args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-5)
    main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT for text classification")

    # Add arguments for flexibility
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pretrained BERT model name")
    parser.add_argument("--dataset_split", type=str, default="train",
                        help="Dataset split to use (train/test/validation)")
    parser.add_argument("--num_labels", type=int, default=5, help="Number of output labels for classification")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save checkpoints and logs")
    parser.add_argument("--model_output_dir", type=str, default="./scientific_bert",
                        help="Directory to save final model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer")
    parser.add_argument("--logging_steps", type=int, default=100, help="Number of steps between logging")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Maximum number of checkpoints to keep")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of CPU workers for data processing")
    parser.add_argument("--fp16", action='store_true', help="Enable mixed precision training for GPUs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Steps for gradient accumulation")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation steps during training")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory to save TensorBoard logs")

    args = parser.parse_args()

    # Hyperparameter optimization using Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

