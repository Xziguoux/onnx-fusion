"""
Constant folding optimization for Enhanced ONNX Tool.
"""

from typing import Dict, List, Set, Optional

from enhanced_onnx_tool.graph.graph import Graph
from enhanced_onnx_tool.utils.logging import get_logger

logger = get_logger(__name__)

def constant_folding(graph: Graph) -> int:
    """
    Perform constant folding optimization on the graph.
    
    Args:
        graph: Computation graph
        
    Returns:
        Number of nodes removed
    """
    # Identify constant tensors
    constant_tensors = _identify_constant_tensors(graph)
    
    # Identify foldable nodes
    foldable_nodes = _identify_foldable_nodes(graph, constant_tensors)
    
    if not foldable_nodes:
        logger.info("No foldable nodes found")
        return 0
    
    logger.info(f"Found {len(foldable_nodes)} foldable nodes")
    
    # Fold nodes
    removed_count = 0
    for node in foldable_nodes:
        if _fold_node(graph, node, constant_tensors):
            removed_count += 1
    
    # Rebuild graph structure
    graph._build_graph_structure()
    
    logger.info(f"Removed {removed_count} nodes by constant folding")
    return removed_count

def _identify_constant_tensors(graph: Graph) -> Set[str]:
    """
    Identify constant tensors in the graph.
    
    Args:
        graph: Computation graph
        
    Returns:
        Set of constant tensor names
    """
    constant_tensors = set(graph.initializers)
    
    # Tensors with no producers (but not inputs) are constants
    for name, tensor in graph.tensor_map.items():
        if (name not in graph.inputs and 
            name not in graph._producers and 
            tensor.is_constant):
            constant_tensors.add(name)
    
    return constant_tensors

def _identify_foldable_nodes(graph: Graph, constant_tensors: Set[str]) -> List[Node]:
    """
    Identify nodes that can be folded.
    
    Args:
        graph: Computation graph
        constant_tensors: Set of constant tensor names
        
    Returns:
        List of foldable nodes
    """
    foldable_nodes = []
    
    for node in graph.nodes:
        # Skip nodes with no inputs
        if not node.inputs:
            continue
        
        # Check if all inputs are constant
        if all(name in constant_tensors for name in node.inputs):
            # Avoid folding nodes that are used for shape inference
            if not _is_shape_inference_node(node):
                foldable_nodes.append(node)
    
    return foldable_nodes

def _is_shape_inference_node(node: Node) -> bool:
    """
    Check if a node is used for shape inference.
    
    Args:
        node: Node to check
        
    Returns:
        True if node is used for shape inference
    """
    # Shape-related operators that should not be folded
    shape_ops = {'Shape', 'Size', 'NonZero', 'ConstantOfShape'}
    
    return node.op_type in shape_ops

def _fold_node(graph: Graph, node: Node, constant_tensors: Set[str]) -> bool:
    """
    Fold a constant node.
    
    Args:
        graph: Computation graph
        node: Node to fold
        constant_tensors: Set of constant tensor names
        
    Returns:
        True if node was folded
    """
    try:
        # Get input tensors
        input_tensors = graph.get_node_inputs(node)
        
        # Compute output values
        output_values = node.value_infer(input_tensors)
        
        # Check if value inference succeeded
        if not all(value is not None for value in output_values):
            logger.debug(f"Value inference failed for node {node.name}")
            return False
# Update output tensors with computed values
        for i, output_name in enumerate(node.outputs):
            if i < len(output_values):
                tensor = graph.tensor_map[output_name]
                tensor.numpy = output_values[i]
                
                # Add to constant tensors set
                constant_tensors.add(output_name)
                
                logger.debug(f"Folded node {node.name}, output {output_name}")
        
        # Remove the folded node
        graph.remove_node(node)
        
        return True
    
    except Exception as e:
        logger.debug(f"Failed to fold node {node.name}: {str(e)}")
        return False
