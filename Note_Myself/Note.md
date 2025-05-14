Graph Agentic Network (GAN) Framework Overview
The GAN framework introduces a novel approach to graph learning where each node functions as an autonomous agent powered by large language models (LLMs) for decision-making. This replaces traditional message passing in graph neural networks with agent-based decision making.
Core Components

NodeState (node.py)

Stores the internal state of each node agent including:

Node ID, features, hidden state
Current and predicted labels
Message queue for communication
Memory for action history


Provides methods to manage messages and track layers


NodeAgent (node.py)

Represents individual nodes as autonomous agents
Maintains a state and communicates with the LLM interface
Core methods:

step(): Makes decisions and executes actions
receive_message(): Handles incoming messages
_prepare_context(): Formats node state for LLM decision-making
_create_action(): Converts LLM decisions into concrete actions




Actions (actions.py)

Base Action class with several specialized subclasses:

RetrieveAction: Gets information from neighboring nodes
BroadcastAction: Sends messages to selected neighbors
UpdateAction: Updates the node's own state
NoOpAction: Does nothing




AgenticGraph (graph.py)

Represents the entire graph structure with node agents
Initializes nodes from adjacency matrix and features
Provides methods to:

Access nodes and their neighbors
Run computation layers
Process nodes in batches




GraphAgenticNetwork (graph.py)

Main class that orchestrates the entire framework
Manages forward passes through multiple layers
Retrieves predictions and node representations
Supports batch processing for scalability


LLMInterface (llm.py)

Communicates with LLMs for agent decision-making
Formats prompts for action decisions
Parses LLM responses into concrete actions
Provides a MockLLMInterface for testing


Utilities (utils.py)

Evaluation metrics for node classification
Visualization functions for node embeddings and graphs
Model comparison tools
Random seed setting for reproducibility



Workflow

Initialization:

Create a GraphAgenticNetwork with graph data
Initialize each node as a NodeAgent with its own state


Forward Pass:

For each layer:

Each node agent prepares its context
The LLM decides actions based on this context
The agent executes these actions
Nodes update for the next layer




Actions:

Nodes can retrieve information from neighbors
Nodes can broadcast messages to neighbors
Nodes can update their own state
Decisions on which actions to take are made by the LLM


Prediction:

After all layers, node representations are collected
These can be used for downstream tasks like classification



Key Features

LLM-powered decision making: Each node uses an LLM to decide what actions to take
Selective communication: Nodes choose which neighbors to interact with
Autonomous agency: Nodes independently decide when to retrieve, broadcast, or update
Memory: Nodes maintain history of actions and interactions
Batched processing: Support for processing nodes in batches for scalability
Visualization tools: Functions to visualize node embeddings and graph structure


* 
hidden state problem
目前无法传递
我希望hidden state也是可以被传播的（retrieve 或 broadcast）,具体的选择交给llm（避免过度传播）

* 
🔧 如果节点没有邻居，LLM 只能靠 prompt 中提供的信息来推理；否则它什么也不知道。
所以你可以：
在 _prepare_context() 中判断邻居数为 0 时，显式在 prompt 中加提示；
甚至提供几个“典型节点”的特征 + 标签供 LLM 参考（类似 few-shot）；

目前好像这几个方法都不完美，我的目标是全智能，few-shot的话好像就又回到需要先了解数据集的情况了，当然如果能让LLM自动选择合适的典型节点和标签告诉无邻居节点，那也是fully-auto的！不过其实我的本意是先绕过这个问题，先解决TAG（text-attributed graph）的情形，因为可以直接通过text description 来判断节点类别或边是否相连，不知道为什么下载的这个arkiv并不是TAG数据集。不过也没事如果这个embedding情形都解决了那TAG更不在话下

一个粗略的想法是，node如果发现自己没有邻居，那么在下一layer就可以在prompt中请求获取特征节点+标签
以及，更常见的情况，发现自己没标签，邻居也没标签，那就需要等待邻居获得更远端的标签信息


✅ 当前先调通 embedding-only 场景（如 OGBN-Arxiv）；

🎯 目标是实现全智能、自主学习与判断，无需 few-shot 教导；

🧠  TAG（Text-Attributed Graph），利用文本 prompt 提供原生可解释性；

📌 对无邻居节点的分类问题，你希望未来由 LLM 自主选择典型节点并进行知识“广播”或迁移。

我们之后可以基于这个目标继续往前推，比如设计：

tag-aware 数据集加载机制；

node embedding + text prompting 融合路径；

“典型节点选择 + label summary” 的自洽传播方式。