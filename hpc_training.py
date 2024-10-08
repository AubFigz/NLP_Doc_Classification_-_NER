import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import logging
import argparse
import os
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def setup_device_and_distributed(args):
    """
    Setup device for GPU or CPU training, and initialize distributed training if needed.
    """
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Using device: {device}")
    return device


def load_model(device, model_name="bert-base-uncased", num_labels=2):
    """
    Load pre-trained BERT model for classification.
    """
    logging.info(f"Loading BERT model {model_name} with {num_labels} labels.")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    return model


def load_data_loader(tokenized_datasets, batch_size, distributed):
    """
    Load the DataLoader with optional distributed sampling for multi-GPU training.
    """
    logging.info(f"Loading DataLoader with batch size {batch_size}.")
    sampler = DistributedSampler(tokenized_datasets) if distributed else None
    return DataLoader(tokenized_datasets, batch_size=batch_size, sampler=sampler)


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir="checkpoints"):
    """
    Save model checkpoints to ensure training progress is preserved.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)
    logging.info(f"Checkpoint saved at {checkpoint_path}")


def train(args):
    # Setup device and distributed training
    device = setup_device_and_distributed(args)

    # Load model
    model = load_model(device, args.model_name, args.num_labels)

    # Setup DataLoader
    train_loader = load_data_loader(args.tokenized_datasets, args.batch_size, args.distributed)

    # Setup optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Optional mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

    # Training loop
    logging.info("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(input_ids=batch['input_ids'].to(device), labels=batch['labels'].to(device))
                loss = outputs.loss
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()  # Update learning rate
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        logging.info(f"Epoch {epoch + 1}/{args.epochs} completed. Average Loss: {total_loss / len(train_loader)}")

        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, epoch, total_loss)

    logging.info("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT model with HPC and Distributed Training")

    # Add arguments for flexibility
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pre-trained model name")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of classification labels")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoader")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for optimizer")
    parser.add_argument("--distributed", action='store_true', help="Enable distributed training")
    parser.add_argument("--fp16", action='store_true', help="Enable mixed-precision training")
    parser.add_argument("--tokenized_datasets", type=str, required=True, help="Path to tokenized dataset")

    args = parser.parse_args()

    # Load tokenized datasets from file
    args.tokenized_datasets = torch.load(args.tokenized_datasets)

    train(args)
