# Graph_Agentic_Network
Node is Agent!

graph_agentic_network/
├── README.md                 # Project documentation
├── setup.sh                  # Environment setup script
├── download_data.sh          # Dataset download script
├── requirements.txt          # Python dependencies
├── gan/
│   ├── __init__.py
│   ├── actions.py            # Action classes for node agents
│   ├── llm.py                # LLM interface for decision making
│   ├── node.py               # Node agent implementation
│   ├── graph.py              # Graph and network implementation
│   └── utils.py              # Utility functions
├── data/
│   └── dataset.py            # Data loading utilities
├── baselines/
│   ├── __init__.py
│   └── gcn.py                # GCN baseline implementation
├── experiments/
│   ├── __init__.py
│   ├── node_classification.py # Node classification experiment
│   └── visualize.py          # Visualization utilities
└── config.py                 # Configuration parameters