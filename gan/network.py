class GraphAgenticNetwork:
    """Graph agentic network implementation."""

    def __init__(
        self,
        adj_matrix: torch.Tensor,
        node_texts: Dict[int, str],
        llm_interface: 'LLMInterface',
        labels: Optional[torch.Tensor] = None,
        train_idx: Optional[torch.Tensor] = None,
        num_layers: int = config.NUM_LAYERS
    ):
        """
        Initialize the graph agentic network.
        
        Args:
            adj_matrix: Adjacency matrix of the graph
            node_texts: Dictionary mapping node IDs to their text descriptions
            llm_interface: Interface to the large language model
            labels: Optional tensor of node labels
            train_idx: Optional tensor of training node indices
            num_layers: Number of message passing layers
        """
        self.graph = AgenticGraph(adj_matrix, llm_interface, labels, train_idx)
        self.num_layers = num_layers 