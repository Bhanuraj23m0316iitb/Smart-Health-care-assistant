# Base image
FROM python:3.10-slim

# Avoid Python writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Set writeable vectorstore directory for Hugging Face Spaces
ENV VECTORSTORE_DIR="/tmp/vector_db_dir"

# Expose Streamlit port
EXPOSE 8501

# Run vectorization first, then launch app
CMD ["sh", "-c", "python vectorize_documents.py && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
