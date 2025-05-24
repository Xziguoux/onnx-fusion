"""
Compute Engine for Enhanced ONNX Tool.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Set, Union

from enhanced_onnx_tool.graph.graph import Graph
from enhanced_onnx_tool.core.tensor import Tensor
from enhanced_onnx_tool.utils.logging import get_logger

logger = get_logger(__name__)

class ComputeEngine:
    """Computation engine for ONNX graphs."""
    
    def __init__(self):
        """Initialize compute engine."""
        self.tensor_values = {}  # Map from tensor name to value
    
    def execute(self, 
               graph: Graph, 
               inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Execute the graph with the given inputs.
        
        Args:
            graph: Computation graph
            inputs: Dictionary mapping input names to numpy arrays
            
        Returns:
            Dictionary mapping output names to numpy arrays
        """
        # Initialize tensor values
        self.tensor_values = {}
        
        # Add inputs
        for name, array in inputs.items():
            if name not in graph.inputs:
                raise ValueError(f"Unknown input tensor: {name}")
            self.tensor_values[name] = array
        
        # Add initializers
        for name in graph.initializers:
            tensor = graph.tensor_map[name]
            if tensor.is_constant and tensor.numpy is not None:
                self.tensor_values[name] = tensor.numpy
        
        # Execute nodes in topological order
        for node in graph.nodes:
            try:
                self._execute_node(graph, node)
            except Exception as e:
                logger.error(f"Execution failed for node {node.name} ({node.op_type}): {str(e)}")
                raise RuntimeError(f"Execution failed for node {node.name} ({node.op_type}): {str(e)}")
        
        # Collect outputs
        outputs = {}
        for name in graph.outputs:
            if name in self.tensor_values:
                outputs[name] = self.tensor_values[name]
            else:
                logger.warning(f"Output tensor '{name}' not computed")
        
        return outputs
    
    def _execute_node(self, graph: Graph, node: Node) -> None:
        """
        Execute a single node.
        
        Args:
            graph: Computation graph
            node: Node to execute
        """
        # Check if all inputs are available
        for name in node.inputs:
            if name not in self.tensor_values:
                logger.warning(f"Input tensor '{name}' not available for node {node.name}")
                return
        
        # Get input tensors
        input_tensors = []
        for name in node.inputs:
            tensor = Tensor(name)
            tensor.numpy = self.tensor_values[name]
            input_tensors.append(tensor)
        
        # Execute node
        try:
            output_values = node.value_infer(input_tensors)
            
            # Store output values
            for i, name in enumerate(node.outputs):
                if i < len(output_values) and output_values[i] is not None:
                    self.tensor_values[name] = output_values[i]
        
        except Exception as e:
            logger.error(f"Execution failed for node {node.name} ({node.op_type}): {str(e)}")
            raise
    
    def profile_memory(self, graph: Graph) -> Dict[str, Any]:
        """
        Profile memory usage of the graph.
        
        Args:
            graph: Computation graph
            
        Returns:
            Dictionary with memory profiling results
        """
        # Ensure all tensors have shapes
        if not all(tensor.shape for tensor in graph.tensor_map.values()):
            logger.warning("Some tensors have unknown shapes, memory profiling may be inaccurate")
        
        # Initialize metrics
        total_weights = 0
        total_activations = 0
        max_activation_size = 0
        per_node_memory = {}
        
        # Track live tensors at each step
        live_tensors = set(graph.inputs + graph.initializers)
        tensor_sizes = {}
        
        # Calculate tensor sizes
        for name, tensor in graph.tensor_map.items():
            if tensor.shape:
                size = np.prod(tensor.shape) * 4  # Assume float32 (4 bytes)
                tensor_sizes[name] = size
                
                # Add weights memory
                if name in graph.initializers:
                    total_weights += size
        
        # Process nodes in topological order
        for node in graph.nodes:
            # Node's inputs are consumed, outputs are produced
            input_sizes = [tensor_sizes.get(name, 0) for name in node.inputs if name not in graph.initializers]
            output_sizes = [tensor_sizes.get(name, 0) for name in node.outputs]
            
            # Update live tensors
            for name in node.inputs:
                if name in live_tensors and name not in graph.initializers:
                    # Remove consumed tensors
                    consumers = graph._consumers.get(name, [])
                    if all(consumer_name in per_node_memory for consumer_name in consumers):
                        live_tensors.remove(name)
            
            for name in node.outputs:
                # Add produced tensors
                live_tensors.add(name)
            
            # Calculate current activation memory
            current_activation_size = sum(tensor_sizes.get(name, 0) for name in live_tensors if name not in graph.initializers)
            max_activation_size = max(max_activation_size, current_activation_size)
            
            # Store per-node memory
            per_node_memory[node.name] = {
                'op_type': node.op_type,
                'input_sizes': input_sizes,
                'output_sizes': output_sizes,
                'activation_size': current_activation_size
            }
            
            # Add to total activations
            total_activations += sum(output_sizes)
        
        return {
            'total_weights': total_weights,
            'total_activations': total_activations,
            'max_activation_size': max_activation_size,
            'per_node': per_node_memory
        }
