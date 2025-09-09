# Database Dockerfile for Fraud Detection Chatbot
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for data processing and API
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pandas flask flask-cors

# Copy database initialization script and API
COPY init_database.py .
COPY database_api.py .

# Create directories
RUN mkdir -p /app/data /app/logs

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash dbuser && \
    chown -R dbuser:dbuser /app
USER dbuser

# Expose port for database API
EXPOSE 5432

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5432/health || exit 1

# Default command - initialize database then start API
CMD ["sh", "-c", "python init_database.py && python database_api.py"]
