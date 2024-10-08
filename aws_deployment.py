import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import logging
import argparse
from tqdm import tqdm

# Set up logging for better debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def upload_model_s3(local_model_path, bucket_name, s3_model_path):
    """
    Upload a file to an S3 bucket with enhanced error handling and progress bar.
    :param local_model_path: Local path to the model file
    :param bucket_name: Name of the S3 bucket
    :param s3_model_path: S3 path to upload the file
    """
    s3 = boto3.client('s3')

    # Check if the file exists before uploading
    if not os.path.isfile(local_model_path):
        logging.error(f"File not found: {local_model_path}")
        return

    # Get the file size for progress tracking
    file_size = os.path.getsize(local_model_path)

    try:
        logging.info(f"Uploading {local_model_path} to S3 bucket {bucket_name} at {s3_model_path}...")

        # Use a progress bar for the upload
        with tqdm(total=file_size, unit='B', unit_scale=True,
                  desc=f"Uploading {os.path.basename(local_model_path)}") as pbar:
            s3.upload_file(
                local_model_path,
                bucket_name,
                s3_model_path,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
            )

        logging.info(f"Model uploaded successfully to S3 at {s3_model_path}")
    except NoCredentialsError:
        logging.error("AWS credentials not found. Make sure they are configured properly.")
    except ClientError as e:
        logging.error(f"Client error during upload: {e}")
    except Exception as e:
        logging.error(f"Error uploading model: {e}")


def main(args):
    """
    Main function to handle S3 upload based on provided arguments.
    """
    upload_model_s3(args.local_model_path, args.bucket_name, args.s3_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a model to an AWS S3 bucket")

    # Command-line arguments for flexibility
    parser.add_argument("--local_model_path", type=str, required=True, help="Local path to the model file")
    parser.add_argument("--bucket_name", type=str, required=True, help="S3 bucket name")
    parser.add_argument("--s3_model_path", type=str, required=True, help="S3 path for the uploaded model")

    args = parser.parse_args()

    # Run the main upload function with the provided arguments
    main(args)
