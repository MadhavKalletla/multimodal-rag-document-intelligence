# Base image with Python 3.11
FROM python:3.11-slim

# Prevent python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies including Tesseract OCR and Poppler utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for caching layers
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and other project components
COPY src/ ./src/
COPY tests/ ./tests/

# Create directory for persistent vector store
RUN mkdir -p vector_store && chmod 777 vector_store

# Expose port 7860 for HuggingFace Spaces compatibility
EXPOSE 7860

# CMD to run the streamlit app on the specified port
CMD ["streamlit", "run", "src/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
