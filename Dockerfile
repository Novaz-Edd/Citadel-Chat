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

# Create data directories
RUN mkdir -p citadel_vault citadel_memory

# Expose port (Render uses PORT env var)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/docs || exit 1

# Start the application
CMD ["python", "main.py"]
