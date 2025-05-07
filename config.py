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
LLM_BACKEND = "remote"  # Options: "remote", "mock", "flan-local"
LLM_MODEL = "Qwen2.5-14B-Instruct" # "Qwen2.5-14B-Instruct" # "Qwen2.5-14B-Instruct" # "llama-3.1-8b-instruct" or "google/flan-t5-xl"
REMOTE_LLM_ENDPOINT = "http://localhost:8001/v1/chat/completions" #"http://localhost:8001/v1/completions"

# Experiment settings
RANDOM_SEED = 42
NUM_LAYERS = 3  # Number of GAN layers to run
BATCH_SIZE = 16  # Batch size for processing nodes
MAX_NEIGHBORS = 30  # Maximum number of neighbors to consider

# GCN baseline settings
GCN_HIDDEN_DIM = 256
GCN_NUM_LAYERS = 2
GCN_DROPOUT = 0.5
GCN_LEARNING_RATE = 0.01
GCN_WEIGHT_DECAY = 5e-4
GCN_EPOCHS = 200

# GAT baseline settings
GAT_HIDDEN_DIM = 256
GAT_NUM_LAYERS = 2
GAT_DROPOUT = 0.6
GAT_LEARNING_RATE = 0.005
GAT_WEIGHT_DECAY = 5e-4
GAT_EPOCHS = 200

# GraphSAGE baseline settings
SAGE_HIDDEN_DIM = 256
SAGE_NUM_LAYERS = 2
SAGE_DROPOUT = 0.5
SAGE_LEARNING_RATE = 0.01
SAGE_WEIGHT_DECAY = 5e-4
SAGE_EPOCHS = 200

# Dataset settings
DATASET_NAME = "cora"  # Default dataset

# Task type
TASK_TYPE = "node_classification"  # or "link_prediction"

# Baseline GNN model type
BASELINE_TYPES = ["GCN", "GAT", "GraphSAGE"]  # "GCN", "GAT", "GraphSAGE"

# Agent control settings
ALLOW_TEST_BROADCAST = True  # Allow test nodes to broadcast predicted labels
ALLOW_FALLBACK_UPDATE = True  # Enable fallback label update mechanism
SHOW_LABEL_LIST_IN_PROMPT = True  # Whether to show label names in LLM prompt

# Logging
VERBOSE = True
LOG_INTERVAL = 10

# Debug settings
DEBUG_LLM = True    #  Whether to print the prompt and response for debugging
DEBUG_STEP_SUMMARY = True  # 是否打印每个节点每层的 step summary
DEBUG_MESSAGE_TRACE = True       # 打印消息传递详情（仅在 retrieve / broadcast 时）
DEBUG_FORCE_FALLBACK = False  # 添加 DEBUG_FORCE_FALLBACK 变量

# vllm settings
MAX_MODEL_LEN = 4096
MEMORY_MAX_WORDS = 80
USE_TOKEN_TRUNCATE = not LLM_MODEL.lower().startswith("deepseek")

