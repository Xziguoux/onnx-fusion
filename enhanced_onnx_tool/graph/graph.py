"""
Graph class for Enhanced ONNX Tool.
"""

import onnx
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from collections import defaultdict
import networkx as nx

from enhanced_onnx_tool.core.node import Node
from enhanced_onnx_tool.core.tensor import Tensor
from enhanced_onnx_tool.utils.logging import get_logger

logger = get_logger(__name__)

class Graph:
    """Represents a computation graph, manages nodes and tensors."""
    
    def __init__(self):
        """Initialize an empty graph."""
        self.nodes: List[Node] = []  # Nodes in topological order
        self.node_map: Dict[str, Node] = {}  # Map from node name to Node object
        self.tensor_map: Dict[str, Tensor] = {}  # Map from tensor name to Tensor object
        self.inputs: List[str] = []  # Names of input tensors
        self.outputs: List[str] = []  # Names of output tensors
        self.initializers: List[str] = []  # Names of initializer tensors
        
        # Cached graph structure for traversal
        self._predecessors: Dict[str, List[str]] = defaultdict(list)  # node_name -> [predecessor_names]
        self._successors: Dict[str, List[str]] = defaultdict(list)  # node_name -> [successor_names]
        self._consumers: Dict[str, List[str]] = defaultdict(list)  # tensor_name -> [consumer_node_names]
        self._producers: Dict[str, Optional[str]] = {}  # tensor_name -> producer_node_name
    
    @classmethod
    def from_onnx(cls, model: onnx.ModelProto) -> 'Graph':
        """
        Create a Graph from an ONNX ModelProto.
        
        Args:
            model: ONNX ModelProto object
            
        Returns:
            New Graph instance
        """
        graph = cls()
        
        # Process graph inputs
        for input_proto in model.graph.input:
            name = input_proto.name
            graph.inputs.append(name)
            tensor = Tensor.from_onnx(input_proto)
            graph.tensor_map[name] = tensor
        
        # Process initializers
        for initializer in model.graph.initializer:
            name = initializer.name
            graph.initializers.append(name)
            tensor = Tensor.from_onnx_tensor(initializer)
            graph.tensor_map[name] = tensor
            # Mark as not an input if it's also in the inputs list
            if name in graph.inputs:
                graph.inputs.remove(name)
        
        # Process outputs
        for output_proto in model.graph.output:
            name = output_proto.name
            graph.outputs.append(name)
            if name not in graph.tensor_map:
                tensor = Tensor.from_onnx(output_proto)
                graph.tensor_map[name] = tensor
        
        # Process nodes
        for node_proto in model.graph.node:
            node = Node.from_onnx(node_proto)
            graph.add_node(node)
        
        # Ensure topological ordering
        graph.topological_sort()
        
        # Build graph structure caches
        graph._build_graph_structure()
        
        return graph
    
    def to_onnx(self, model: Optional[onnx.ModelProto] = None) -> onnx.ModelProto:
        """
        Convert the Graph back to an ONNX ModelProto.
        
        Args:
            model: Optional original model to update (to preserve metadata)
            
        Returns:
            ONNX ModelProto
        """
        if model is None:
            # Create a new model
            model = onnx.ModelProto()
            model.ir_version = 7
            model.producer_name = "enhanced_onnx_tool"
            model.opset_import.extend([onnx.helper.make_opsetid("", 13)])
            model.graph = onnx.GraphProto()
        
        # Clear existing graph content
        del model.graph.node[:]
        del model.graph.input[:]
        del model.graph.output[:]
        del model.graph.initializer[:]
        
        # Add inputs
        for name in self.inputs:
            tensor = self.tensor_map[name]
            input_proto = tensor.to_onnx_value_info()
            model.graph.input.append(input_proto)
        
        # Add initializers
        for name in self.initializers:
            tensor = self.tensor_map[name]
            initializer = tensor.to_onnx_tensor()
            model.graph.initializer.append(initializer)
        
        # Add outputs
        for name in self.outputs:
            tensor = self.tensor_map[name]
            output_proto = tensor.to_onnx_value_info()
            model.graph.output.append(output_proto)
        
        # Add nodes
        for node in self.nodes:
            node_proto = node.to_onnx()
            model.graph.node.append(node_proto)
        
        # Value infos for intermediate tensors with shape info
        for name, tensor in self.tensor_map.items():
            if name not in self.inputs and name not in self.outputs and name not in self.initializers:
                if tensor.shape:  # Only add if shape is known
                    value_info = tensor.to_onnx_value_info()
                    model.graph.value_info.append(value_info)
        
        return model
    
    def add_node(self, node: Node) -> Node:
        """
        Add a node to the graph.
        
        Args:
            node: Node to add
            
        Returns:
            Added node
        """
        # Check for duplicate name
        if node.name in self.node_map:
            raise ValueError(f"Node with name '{node.name}' already exists in the graph")
        
        # Add node to collections
        self.nodes.append(node)
        self.node_map[node.name] = node
        
        # Create or update tensor references
        for input_name in node.inputs:
            if input_name not in self.tensor_map:
                # Create tensor if it doesn't exist
                self.tensor_map[input_name] = Tensor(input_name)
            # Update tensor consumers
            self._consumers[input_name].append(node.name)
        
        for output_name in node.outputs:
            if output_name not in self.tensor_map:
                # Create tensor if it doesn't exist
                self.tensor_map[output_name] = Tensor(output_name)
            # Update tensor producer
            self._producers[output_name] = node.name
        
        # Update graph structure caches
        self._update_graph_structure_for_node(node)
        
        return node
    
    def remove_node(self, node_or_name: Union[Node, str]) -> None:
        """
        Remove a node from the graph.
        
        Args:
            node_or_name: Node or node name to remove
        """
        # Get node object
        if isinstance(node_or_name, str):
            if node_or_name not in self.node_map:
                raise ValueError(f"Node '{node_or_name}' not found in graph")
            node = self.node_map[node_or_name]
        else:
            node = node_or_name
            if node.name not in self.node_map:
                raise ValueError(f"Node '{node.name}' not found in graph")
        
        # Remove node from collections
        self.nodes.remove(node)
        del self.node_map[node.name]
        
        # Update tensor references
        for input_name in node.inputs:
            if node.name in self._consumers[input_name]:
                self._consumers[input_name].remove(node.name)
        
        for output_name in node.outputs:
            if self._producers.get(output_name) == node.name:
                self._producers[output_name] = None
        
        # Update graph structure caches
        self._remove_node_from_graph_structure(node)
    
    def topological_sort(self) -> None:
        """Sort the nodes in topological order."""
        # Build a directed graph
        dg = nx.DiGraph()
        
        # Add nodes
        for node in self.nodes:
            dg.add_node(node.name)
        
        # Add edges based on tensor dependencies
        for node in self.nodes:
            for input_name in node.inputs:
                producer = self._producers.get(input_name)
                if producer and producer in self.node_map:
                    dg.add_edge(producer, node.name)
        
        try:
            # Compute topological sort
            sorted_names = list(nx.topological_sort(dg))
            
            # Reorder nodes based on sort
            self.nodes = [self.node_map[name] for name in sorted_names if name in self.node_map]
        except nx.NetworkXUnfeasible:
            logger.warning("Graph contains cycles, topological sort not possible")
    
    def get_node_inputs(self, node: Node) -> List[Tensor]:
        """
        Get the input tensors for a node.
        
        Args:
            node: Node to get inputs for
            
        Returns:
            List of input tensors
        """
        return [self.tensor_map[name] for name in node.inputs if name in self.tensor_map]
    
    def get_node_outputs(self, node: Node) -> List[Tensor]:
        """
        Get the output tensors for a node.
        
        Args:
            node: Node to get outputs for
            
        Returns:
            List of output tensors
        """
        return [self.tensor_map[name] for name in node.outputs if name in self.tensor_map]
    
    def remove_dangling_nodes(self) -> int:
        """
        Remove nodes that don't contribute to any output.
        
        Returns:
            Number of nodes removed
        """
        # Mark nodes that contribute to outputs
        contributing = set()
        
        # Start with output nodes
        queue = []
        for output_name in self.outputs:
            if output_name in self._producers and self._producers[output_name]:
                queue.append(self._producers[output_name])
        
        # Breadth-first search to mark all contributing nodes
        while queue:
            node_name = queue.pop(0)
            if node_name in contributing:
                continue
            
            contributing.add(node_name)
            
            # Add predecessors to queue
            for pred_name in self._predecessors.get(node_name, []):
                if pred_name not in contributing:
                    queue.append(pred_name)
        
        # Remove non-contributing nodes
        to_remove = []
        for node in self.nodes:
            if node.name not in contributing:
                to_remove.append(node)
        
        # Remove nodes
        for node in to_remove:
            self.remove_node(node)
        
        return len(to_remove)
    
    def remove_redundant_nodes(self) -> int:
        """
        Remove redundant nodes like Identity operators.
        
        Returns:
            Number of nodes removed
        """
        # Identify redundant nodes
        redundant = []
        for node in self.nodes:
            # Check if node is an Identity-like operator
            if node.is_identity_like():
                redundant.append(node)
        
        # Remove and reconnect redundant nodes
        removed_count = 0
        for node in redundant:
            # Only handle single-input, single-output nodes
            if len(node.inputs) == 1 and len(node.outputs) == 1:
                input_name = node.inputs[0]
                output_name = node.outputs[0]
                
                # Update consumers of the output tensor
                for consumer_name in self._consumers.get(output_name, []):
                    consumer = self.node_map.get(consumer_name)
                    if consumer:
                        # Replace output with input in consumer's inputs
                        for i, name in enumerate(consumer.inputs):
                            if name == output_name:
                                consumer.inputs[i] = input_name
                        
                        # Update consumer relationships
                        self._consumers[input_name].append(consumer_name)
                
                # Remove the node
                self.remove_node(node)
                removed_count += 1
        
        # Rebuild graph structure
        self._build_graph_structure()
        
        return removed_count
    
    def _build_graph_structure(self) -> None:
        """Build the graph structure caches."""
        # Clear existing caches
        self._predecessors = defaultdict(list)
        self._successors = defaultdict(list)
        self._consumers = defaultdict(list)
        self._producers = {}
        
        # Build tensor producer/consumer relationships
        for node in self.nodes:
            # Outputs: this node produces these tensors
            for output_name in node.outputs:
                self._producers[output_name] = node.name
            
            # Inputs: this node consumes these tensors
            for input_name in node.inputs:
                self._consumers[input_name].append(node.name)
        
        # Build node predecessor/successor relationships
        for node in self.nodes:
            # Predecessors: nodes that produce this node's inputs
            for input_name in node.inputs:
                producer = self._producers.get(input_name)
                if producer and producer != node.name:  # Avoid self-loops
                    self._predecessors[node.name].append(producer)
            
            # Successors: nodes that consume this node's outputs
            for output_name in node.outputs:
                for consumer in self._consumers.get(output_name, []):
                    if consumer != node.name:  # Avoid self-loops
                        self._successors[node.name].append(consumer)
    
    def _update_graph_structure_for_node(self, node: Node) -> None:
        """
        Update graph structure caches for a single node.
        
        Args:
            node: Node to update structure for
        """
        # Update predecessors
        for input_name in node.inputs:
            producer = self._producers.get(input_name)
            if producer and producer != node.name:  # Avoid self-loops
                self._predecessors[node.name].append(producer)
                self._successors[producer].append(node.name)
        
        # Update successors
        for output_name in node.outputs:
            for consumer in self._consumers.get(output_name, []):
                if consumer != node.name:  # Avoid self-loops
                    self._successors[node.name].append(consumer)
                    self._predecessors[consumer].append(node.name)
    
    def _remove_node_from_graph_structure(self, node: Node) -> None:
        """
        Remove a node from the graph structure caches.
        
        Args:
            node: Node to remove from structure
        """
        # Remove from predecessors/successors
        if node.name in self._predecessors:
            del self._predecessors[node.name]
        
        if node.name in self._successors:
            del self._successors[node.name]
        
        # Remove node from other nodes' predecessors/successors
        for pred_list in self._predecessors.values():
            if node.name in pred_list:
                pred_list.remove(node.name)
        
        for succ_list in self._successors.values():
            if node.name in succ_list:
                succ_list.remove(node.name)
    
    def __repr__(self) -> str:
        """String representation of the graph."""
        return f"Graph(nodes={len(self.nodes)}, tensors={len(self.tensor_map)}, inputs={len(self.inputs)}, outputs={len(self.outputs)})"
