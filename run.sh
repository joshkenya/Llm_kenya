#!/bin/bash

# LLM Kenya Startup Script
# Usage: ./run.sh [mode]
# Modes: chat, web, train

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default mode
MODE=${1:-web}

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Welcome to LLM Kenya!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${YELLOW}Python Version: $PYTHON_VERSION${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install requirements if needed
if [ ! -d "venv/lib" ] || [ -z "$(pip list | grep -i numpy)" ]; then
    echo -e "${YELLOW}Installing requirements...${NC}"
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
fi

echo ""
echo -e "${GREEN}Setup complete!${NC}"
echo ""

# Run based on mode
case $MODE in
    chat)
        echo -e "${GREEN}Starting Interactive Chat...${NC}"
        echo ""
        python3 chat.py
        ;;
    web)
        echo -e "${GREEN}Starting Web UI...${NC}"
        echo -e "${YELLOW}Open your browser and navigate to: http://localhost:5000${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
        echo ""
        python3 app.py
        ;;
    train)
        echo -e "${RED}Error: Training mode is designed for Google Colab${NC}"
        echo -e "${YELLOW}Please use: train_colab.ipynb${NC}"
        exit 1
        ;;
    model-info)
        echo -e "${GREEN}Loading Model Information...${NC}"
        echo ""
        python3 << EOF
from model import TransformerModel, Config
config = Config()
model = TransformerModel(config)
print(f"Model Parameters: {model.total_params:,}")
print(f"Model Parameters (Millions): {model.total_params / 1e6:.2f}M")
print(f"\nConfiguration:")
print(f"  Hidden Size: {config.hidden_size}")
print(f"  Number of Layers: {config.num_hidden_layers}")
print(f"  Attention Heads: {config.num_attention_heads}")
print(f"  Vocab Size: {config.vocab_size}")
print(f"  Max Sequence Length: {config.max_position_embeddings}")
EOF
        ;;
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo ""
        echo "Usage: ./run.sh [mode]"
        echo ""
        echo "Available modes:"
        echo "  chat         - Run interactive CLI chat"
        echo "  web          - Run Flask web UI"
        echo "  train        - (Use train_colab.ipynb on Google Colab)"
        echo "  model-info   - Display model information"
        exit 1
        ;;
esac
