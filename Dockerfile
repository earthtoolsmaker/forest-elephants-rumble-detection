# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0


# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to the PATH
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files for dependency installation
COPY pyproject.toml poetry.lock ./

# This line ensures when packages are installed with Poetry a virtual environment is NOT created first. 
# You’re already in a virtual environment by using a docker image
RUN poetry config virtualenvs.create false

# Copy the rest of the application code
COPY src/ ./src
COPY README.md ./
COPY scripts/ ./scripts
COPY data/08_artifacts/ ./data/08_artifacts

# Install Python dependencies
RUN poetry install --only main

# Run the script as the entrypoint
ENTRYPOINT ["poetry", "run", "python", "./scripts/model/yolov8/predict_raven.py", \
            "--input-dir-audio-filepaths", "./data/08_artifacts/audio/rumbles/", \
            "--output-dir", "./data/05_model_output/yolov8/predict/", \
            "--model-weights-filepath", "./data/08_artifacts/model/rumbles/yolov8/weights/best.pt", \
            "--verbose", \
            "--loglevel", "info"]
