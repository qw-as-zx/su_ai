# Use a lighter Python base image
FROM python:3.12-slim

# Set essential environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libmagic1 \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create application directories
RUN mkdir -p /app/logs /app/temp /app/vector_stores
# RUN mkdir -p /app/logs /app/temp /app/vector_stores \
#     && chmod -R 777 /app/logs /app/temp /app/vector_stores



# Expose port
EXPOSE 8000

# Start the application
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000"]