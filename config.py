"""
Configuration settings for the Graph Agentic Network
"""

import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MULTI_GPU = torch.cuda.device_count() > 1

# LLM settings
LLM_BACKEND = "remote"  # Options: "remote", "mock"
LLM_MODEL = "llama-3.1-8b-instruct"
REMOTE_LLM_ENDPOINT = "http://localhost:8001/v1/chat/completions"


# Experiment settings
RANDOM_SEED = 42
NUM_LAYERS = 2  # Number of GAN layers to run
BATCH_SIZE = 16  # Batch size for processing nodes
MAX_NEIGHBORS = 10  # Maximum number of neighbors to consider

# GCN baseline settings
GCN_HIDDEN_DIM = 256
GCN_NUM_LAYERS = 2
GCN_DROPOUT = 0.5
GCN_LEARNING_RATE = 0.01
GCN_WEIGHT_DECAY = 5e-4
GCN_EPOCHS = 200

# Dataset settings
DATASET_NAME = "ogbn-arxiv"  # Default dataset

# Logging
VERBOSE = True
LOG_INTERVAL = 10

# Debug settings
DEBUG_LLM = True    #  Whether to print the prompt and response for debugging
DEBUG_STEP_SUMMARY = True  # 是否打印每个节点每层的 step summary
DEBUG_MESSAGE_TRACE = True       # 打印消息传递详情（仅在 retrieve / broadcast 时）