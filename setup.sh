#!/bin/bash

# Graph Agentic Network setup script
echo "Setting up environment for Graph Agentic Network..."

# Create conda environment
conda create -n gan python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate gan

# Install PyTorch with CUDA support
# Using CUDA 12.6 compatible version
conda install pytorch pytorch-cuda=12.4 torchaudio pytorch-mutex=1.0=cuda -c pytorch -c nvidia -y

# Install PyTorch Geometric
conda install pyg -c pyg -y

# Install other dependencies
pip install ogb transformers tqdm matplotlib networkx scikit-learn pandas huggingface_hub

# Install development tools
pip install jupyter jupyterlab

# Create data directory
mkdir -p data/raw
mkdir -p data/processed

echo "Environment setup complete. Please activate it with:"
echo "conda activate gan"