"""
Graph Agentic Network (GAN) framework

A novel approach to graph learning where each node functions as an autonomous agent,
powered by large language models for decision-making.
"""

from gan.node import NodeState, NodeAgent
from gan.actions import Action, RetrieveAction, BroadcastAction, UpdateAction
from gan.graph import AgenticGraph, GraphAgenticNetwork
from gan.llm import LLMInterface

__version__ = "0.1.0"
