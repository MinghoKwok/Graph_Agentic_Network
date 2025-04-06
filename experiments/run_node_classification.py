def main():
    # Load data
    adj_matrix = load_adjacency_matrix()
    labels = load_labels()
    node_texts = load_node_texts()
    
    # Load train/val/test indices
    train_idx = torch.from_numpy(np.load(f"data/{config.DATASET_NAME}/train_idx.npy"))
    val_idx = torch.from_numpy(np.load(f"data/{config.DATASET_NAME}/val_idx.npy"))
    test_idx = torch.from_numpy(np.load(f"data/{config.DATASET_NAME}/test_idx.npy"))
    
    # Initialize LLM interface
    llm_interface = LLMInterface()
    
    # Initialize network
    gan = GraphAgenticNetwork(
        adj_matrix=adj_matrix,
        node_texts=node_texts,
        llm_interface=llm_interface,
        labels=labels,
        train_idx=train_idx,
        num_layers=config.NUM_LAYERS
    )
    
    # Run inference
    gan.forward()
    
    # Evaluate
    evaluate_predictions(gan, val_idx, test_idx) 