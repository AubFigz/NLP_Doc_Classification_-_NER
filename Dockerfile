# Use the official Python runtime as a parent image with slim base image for smaller size
FROM python:3.9-slim

# Set environment variables to ensure the application runs in non-interactive and reproducible mode
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOME=/app

# Install essential dependencies for building some Python packages and cleaning up unnecessary files afterward
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR $APP_HOME

# Copy requirements file first to leverage Docker cache if dependencies don’t change
COPY requirements.txt .

# Install Python dependencies in one layer and clean up afterwards to reduce image size
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip

# Copy the current directory contents into the container's working directory
COPY . .

# Ensure that all scripts in the /app directory are executable
RUN chmod +x /app/*

# Set a default command to run the application, but allow flexibility via an entrypoint script
CMD ["python", "app.py"]

# Expose the application’s port (optional, depending on app)
EXPOSE 8000
