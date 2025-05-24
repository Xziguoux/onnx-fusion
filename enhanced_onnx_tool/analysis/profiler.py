"""
Model profiler for Enhanced ONNX Tool.
"""

import numpy as np
from typing import Dict, List, Optional, Any

from enhanced_onnx_tool.graph.graph import Graph
from enhanced_onnx_tool.utils.logging import get_logger

logger = get_logger(__name__)

class ModelProfiler:
    """Model profiler for computational and memory analysis."""
    
    def __init__(self, shape_engine=None, compute_engine=None):
        """
        Initialize model profiler.
        
        Args:
            shape_engine: Optional shape engine instance
            compute_engine: Optional compute engine instance
        """
        self._shape_engine = shape_engine
        self._compute_engine = compute_engine
    
    @property
    def shape_engine(self):
        """Lazy-loaded shape engine."""
        if self._shape_engine is None:
            from enhanced_onnx_tool.engines.shape_engine import ShapeEngine
            self._shape_engine = ShapeEngine()
        return self._shape_engine
    
    @property
    def compute_engine(self):
        """Lazy-loaded compute engine."""
        if self._compute_engine is None:
            from enhanced_onnx_tool.engines.compute_engine import ComputeEngine
            self._compute_engine = ComputeEngine()
        return self._compute_engine
    
    def profile(self, graph: Graph) -> Dict[str, Any]:
        """
        Profile a graph for computational and memory metrics.
        
        Args:
            graph: Computation graph
            
        Returns:
            Dictionary with profiling results
        """
        # Ensure all tensors have shapes
        if not all(tensor.shape for tensor in [graph.tensor_map[name] for name in graph.outputs]):
            logger.warning("Some output tensors have unknown shapes, profiling may be inaccurate")
        
        # Compute metrics
        compute_metrics = self._profile_compute(graph)
        memory_metrics = self.compute_engine.profile_memory(graph)
        
        # Combine metrics
        metrics = {
            'total_macs': compute_metrics['total_macs'],
            'total_params': compute_metrics['total_params'],
            'total_weights': memory_metrics['total_weights'],
            'total_activations': memory_metrics['total_activations'],
            'max_activation_size': memory_metrics['max_activation_size'],
            'per_node': {}
        }
        
        # Combine per-node metrics
        for node_name in compute_metrics['per_node'].keys():
            compute_node = compute_metrics['per_node'].get(node_name, {})
            memory_node = memory_metrics['per_node'].get(node_name, {})
            
            metrics['per_node'][node_name] = {
                'op_type': compute_node.get('op_type', ''),
                'macs': compute_node.get('macs', 0),
                'params': compute_node.get('params', 0),
                'input_shapes': compute_node.get('input_shapes', []),
                'output_shapes': compute_node.get('output_shapes', []),
                'activation_size': memory_node.get('activation_size', 0)
            }
        
        return metrics
    
    def _profile_compute(self, graph: Graph) -> Dict[str, Any]:
        """
        Profile computational metrics of the graph.
        
        Args:
            graph: Computation graph
            
        Returns:
            Dictionary with computational profiling results
        """
        # Initialize metrics
        total_macs = 0
        total_params = 0
        per_node = {}
        
        # Profile each node
        for node in graph.nodes:
            try:
                input_tensors = graph.get_node_inputs(node)
                output_tensors = graph.get_node_outputs(node)
                
                # Call node's profile method
                macs, params = node.profile(input_tensors, output_tensors)
                
                # Update totals
                total_macs += macs
                total_params += params
                
                # Store per-node statistics
                per_node[node.name] = {
                    'op_type': node.op_type,
                    'macs': macs,
                    'params': params,
                    'input_shapes': [t.shape for t in input_tensors],
                    'output_shapes': [t.shape for t in output_tensors]
                }
            except Exception as e:
                logger.warning(f"Failed to profile node {node.name}: {str(e)}")
        
        return {
            'total_macs': total_macs,
            'total_params': total_params,
            'per_node': per_node
        }
    
    def summarize(self, metrics: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of profiling results.
        
        Args:
            metrics: Profiling metrics
            
        Returns:
            Summary string
        """
        lines = []
        
        # Overall metrics
        lines.append("========== Model Profile Summary ==========")
        lines.append(f"Total MACs: {metrics['total_macs']:,}")
        lines.append(f"Total Parameters: {metrics['total_params']:,}")
        lines.append(f"Total Weights Memory: {self._format_bytes(metrics['total_weights'])}")
        lines.append(f"Total Activations Memory: {self._format_bytes(metrics['total_activations'])}")
        lines.append(f"Maximum Activation Memory: {self._format_bytes(metrics['max_activation_size'])}")
        lines.append("")
        
        # Top nodes by MACs
        lines.append("Top 10 nodes by MACs:")
        sorted_nodes = sorted(
            metrics['per_node'].items(), 
            key=lambda x: x[1]['macs'], 
            reverse=True
        )
        for i, (name, stats) in enumerate(sorted_nodes[:10]):
            macs_percent = (stats['macs'] / metrics['total_macs'] * 100) if metrics['total_macs'] > 0 else 0
            lines.append(f"{i+1}. {name} ({stats['op_type']}): {stats['macs']:,} MACs ({macs_percent:.2f}%)")
        lines.append("")
        
        # MACs by operator type
        lines.append("MACs by operator type:")
        op_type_macs = {}
        for _, stats in metrics['per_node'].items():
            op_type = stats['op_type']
            if op_type not in op_type_macs:
                op_type_macs[op_type] = 0
            op_type_macs[op_type] += stats['macs']
        
        sorted_ops = sorted(op_type_macs.items(), key=lambda x: x[1], reverse=True)
        for op_type, macs in sorted_ops:
            macs_percent = (macs / metrics['total_macs'] * 100) if metrics['total_macs'] > 0 else 0
            lines.append(f"{op_type}: {macs:,} MACs ({macs_percent:.2f}%)")
        
        return "\n".join(lines)
    
    def _format_bytes(self, bytes_value: int) -> str:
        """
        Format bytes value to human-readable string.
        
        Args:
            bytes_value: Number of bytes
            
        Returns:
            Formatted string
        """
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        unit_index = 0
        value = bytes_value
        
        while value >= 1024 and unit_index < len(units) - 1:
            value /= 1024
            unit_index += 1
        
        return f"{value:.2f} {units[unit_index]}"
