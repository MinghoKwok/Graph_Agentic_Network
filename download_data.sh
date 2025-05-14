#!/bin/bash

# Download script for datasets
echo "Downloading datasets for Graph Agentic Network..."

# Create data directories if they don't exist
mkdir -p data/raw
mkdir -p data/processed

# OGB-Arxiv dataset will be automatically downloaded by the OGB library
# This script will download any additional resources if needed

# Download pre-trained LLM model (if using a local model)
MODEL_DIR="models"
mkdir -p $MODEL_DIR

# You may uncomment and modify the following lines if you want to download a specific model
# Note: This requires proper authentication for access-restricted models
# huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir $MODEL_DIR/llama-2-7b-chat

# Log completion
echo "Dataset preparation complete!"
echo "Note: The OGB-Arxiv dataset will be automatically downloaded when you first run the experiments"