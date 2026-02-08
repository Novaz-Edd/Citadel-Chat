# ─────────────────────────────────────────────
# Citadel-Chat Dockerfile
# FastAPI + Ollama (all-in-one)
# ─────────────────────────────────────────────
FROM python:3.11-slim

# Install system dependencies (including zstd required by Ollama installer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create data directories
RUN mkdir -p citadel_vault citadel_memory

# Make start script executable
COPY start.sh .
RUN chmod +x start.sh

# Expose port (Render uses PORT env var)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/docs || exit 1

# Start everything
CMD ["./start.sh"]
