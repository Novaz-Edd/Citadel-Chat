# ─────────────────────────────────────────────
# Citadel-Chat Dockerfile
# FastAPI + Groq API (lightweight, no Ollama)
# ─────────────────────────────────────────────
FROM python:3.11-slim

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Data directories are created at runtime by main.py using the DATA_DIR env var.
# On Render, DATA_DIR=/app/data (the persistent disk mount point).
# Locally, they are created next to main.py.

# Expose port (Render uses PORT env var)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/docs || exit 1

# Start the application
CMD ["python", "main.py"]
