#!/bin/bash

echo "=================================="
echo "Taylor Swift Analysis - Agent Setup"
echo "=================================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "✓ Ollama already installed"
fi

# Start Ollama service (if not running)
echo "Starting Ollama service..."
ollama serve > /dev/null 2>&1 &
sleep 2

# Pull memory-efficient models for 16GB RAM MacBook Pro
echo "Pulling models (optimized for 16GB RAM)..."
echo "This will download ~4-5GB total..."
ollama pull llama3.2:3b
ollama pull phi3:mini
ollama pull mxbai-embed-large

# Install Python dependencies
echo "Installing Python dependencies..."
pip install ollama requests

# Test connection
echo "Testing setup..."
python -m src.agents.ollama_client

echo ""
echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "Models installed:"
echo "  - llama3.2:3b (~2GB RAM)"
echo "  - phi3:mini (~2.3GB RAM)"
echo ""
echo "Run the demo with:"
echo "  python demo_agents.py"
echo ""

