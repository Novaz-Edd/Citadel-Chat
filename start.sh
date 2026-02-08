#!/bin/bash
# ─────────────────────────────────────────────
# Citadel-Chat Startup Script
# Sets up data dirs → Starts FastAPI
# ─────────────────────────────────────────────
set -e

echo "🏰 Citadel-Chat: Starting up..."

# Setup persistent data directory
DATA_DIR=${DATA_DIR:-/app}
mkdir -p "$DATA_DIR/citadel_vault" "$DATA_DIR/citadel_memory"

# Start FastAPI application
PORT=${PORT:-8000}
echo "🚀 Starting Citadel-Chat on port $PORT..."
exec python main.py
