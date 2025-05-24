from enhanced_onnx_tool.core.node import Node
"""
Shape Engine for Enhanced ONNX Tool.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Set, Union

from enhanced_onnx_tool.graph.graph import Graph
from enhanced_onnx_tool.core.tensor import Tensor
from enhanced_onnx_tool.utils.logging import get_logger

logger = get_logger(__name__)

class ShapeEngine:
    """Shape inference engine for ONNX graphs."""
    
    def __init__(self):
        """Initialize shape engine."""
        self.dynamic_axes = {}  # Map from tensor name to dynamic axis symbols
    
    def infer(self, 
              graph: Graph, 
              input_tensors: Optional[Dict[str, np.ndarray]] = None) -> Graph:
        """
        Infer shapes for all tensors in the graph.
        
        Args:
            graph: Computation graph
            input_tensors: Dictionary mapping input names to numpy arrays
            
        Returns:
            Graph with inferred shapes
        """
        # Check if inputs have valid shapes
        if not self._check_inputs(graph, input_tensors):
            raise ValueError("Input tensors have invalid shapes. Please provide valid input shapes.")
        
        # Set input tensor values if provided
        if input_tensors:
            for name, array in input_tensors.items():
                if name in graph.tensor_map:
                    graph.tensor_map[name].numpy = array
        
        # Traverse nodes in topological order
        processed = set()
        for node in graph.nodes:
            try:
                self._infer_node_shapes(graph, node, processed)
            except Exception as e:
                logger.error(f"Shape inference failed for node {node.name} ({node.op_type}): {str(e)}")
                raise RuntimeError(f"Shape inference failed for node {node.name} ({node.op_type}): {str(e)}")
        
        return graph
    
    def _check_inputs(self, 
                      graph: Graph, 
                      input_tensors: Optional[Dict[str, np.ndarray]] = None) -> bool:
        """
        Check if input tensors have valid shapes.
        
        Args:
            graph: Computation graph
            input_tensors: Dictionary mapping input names to numpy arrays
            
        Returns:
            True if all inputs are valid
        """
        for name in graph.inputs:
            tensor = graph.tensor_map[name]
            
            # Skip check if input values provided
            if input_tensors and name in input_tensors:
                continue
            
            # Check shape validity
            if not tensor.shape:
                logger.warning(f"Input tensor '{name}' has no shape information")
                return False
            
            # 修改检查逻辑：允许动态维度（-1值）
            # 对于动态维度，我们假设它有效，而不是报错
            # 注释掉或修改下面的检查
            # for i, dim in enumerate(tensor.shape):
            #     if dim <= 0:
            #         logger.warning(f"Input tensor '{name}' has invalid dimension {dim} at axis {i}")
            #         return False
        
        return True
    
    def _infer_node_shapes(self, 
                           graph: Graph, 
                           node: Node, 
                           processed: Set[str]) -> None:
        """
        Infer shapes for a single node.
        
        Args:
            graph: Computation graph
            node: Node to infer shapes for
            processed: Set of already processed tensor names
        """
        # Skip if already processed
        if node.name in processed:
            return
        
        # Check if all input tensors have shapes
        input_tensors = []
        for name in node.inputs:
            if name in graph.tensor_map:
                tensor = graph.tensor_map[name]
                input_tensors.append(tensor)
            else:
                # Create empty tensor if not found
                logger.warning(f"Input tensor '{name}' not found for node {node.name}")
                tensor = Tensor(name)
                graph.tensor_map[name] = tensor
                input_tensors.append(tensor)
        
        # Get output tensors
        output_tensors = []
        for name in node.outputs:
            if name in graph.tensor_map:
                tensor = graph.tensor_map[name]
            else:
                # Create empty tensor if not found
                tensor = Tensor(name)
                graph.tensor_map[name] = tensor
            output_tensors.append(tensor)
        
        # Check if input shapes are known
        missing_shapes = [tensor.name for tensor in input_tensors if not tensor.shape]
        if missing_shapes:
            logger.debug(f"Node {node.name}: missing shapes for inputs {missing_shapes}")
        
        # Try to infer shapes
        try:
            # Call node's shape inference method
            output_shapes = node.shape_infer(input_tensors)
            
            # Update output tensor shapes
            for i, shape in enumerate(output_shapes):
                if i < len(output_tensors) and shape is not None:
                    output_tensors[i].update_shape(shape)
            
            # Mark as processed
            processed.add(node.name)
            
            # Try value inference for constant tensors
            self._try_value_inference(graph, node, input_tensors, output_tensors)
            
        except Exception as e:
            logger.warning(f"Shape inference failed for node {node.name} ({node.op_type}): {str(e)}")
            # Don't add to processed set to allow another attempt if dependencies get resolved
    
    def _try_value_inference(self, 
                            graph: Graph, 
                            node: Node, 
                            input_tensors: List[Tensor], 
                            output_tensors: List[Tensor]) -> None:
        """
        Try to infer values for constant tensors.
        
        Args:
            graph: Computation graph
            node: Node to infer values for
            input_tensors: Input tensors
            output_tensors: Output tensors
        """
        # Check if all input tensors have values and are constant
        all_constant = all(tensor.is_constant for tensor in input_tensors)
        
        if all_constant:
            try:
                # Call node's value inference method
                output_values = node.value_infer(input_tensors)
                
                # Update output tensor values
                for i, value in enumerate(output_values):
                    if i < len(output_tensors) and value is not None:
                        output_tensors[i].numpy = value
                
                logger.debug(f"Value inference succeeded for node {node.name} ({node.op_type})")
            except Exception as e:
                logger.debug(f"Value inference failed for node {node.name} ({node.op_type}): {str(e)}")
