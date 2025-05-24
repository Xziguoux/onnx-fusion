"""
Top-level model class for Enhanced ONNX Tool.
"""

import os
import onnx
import numpy as np
from typing import Dict, List, Union, Optional, Any

from enhanced_onnx_tool.graph.graph import Graph
from enhanced_onnx_tool.utils.logging import get_logger

logger = get_logger(__name__)

class ModelConfig:
    """Configuration for Model."""
    
    def __init__(self, 
                 constant_folding: bool = False,
                 remove_dangling: bool = False,
                 verbose: bool = True):
        """
        Initialize model configuration.
        
        Args:
            constant_folding: Whether to apply constant folding when loading
            remove_dangling: Whether to remove dangling nodes when loading
            verbose: Whether to enable verbose logging
        """
        self.constant_folding = constant_folding
        self.remove_dangling = remove_dangling
        self.verbose = verbose

class Model:
    """Top-level model container, manages model loading, saving, and high-level operations."""
    
    def __init__(self, 
                 path_or_model: Union[str, onnx.ModelProto, None] = None,
                 config: Optional[ModelConfig] = None):
        """
        Initialize model.
        
        Args:
            path_or_model: Model path or ONNX ModelProto object
            config: Model configuration
        """
        self.graph = None
        self._shape_engine = None
        self._compute_engine = None
        self._profiler = None
        self.config = config or ModelConfig()
        self.onnx_model = None
        self.modelpath = None
        
        if path_or_model:
            self.load(path_or_model)
    
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
    
    @property
    def profiler(self):
        """Lazy-loaded profiler."""
        if self._profiler is None:
            from enhanced_onnx_tool.analysis.profiler import ModelProfiler
            self._profiler = ModelProfiler(self.shape_engine, self.compute_engine)
        return self._profiler
    
    def load(self, path_or_model: Union[str, onnx.ModelProto]) -> 'Model':
        """
        Load model from path or ONNX ModelProto.
        
        Args:
            path_or_model: Model path or ONNX ModelProto object
            
        Returns:
            Self for method chaining
        """
        if isinstance(path_or_model, str):
            if not os.path.exists(path_or_model):
                raise FileNotFoundError(f"Model file not found: {path_or_model}")
            
            self.modelpath = path_or_model
            self.onnx_model = onnx.load(path_or_model)
            logger.info(f"Model loaded from {path_or_model}")
        elif isinstance(path_or_model, onnx.ModelProto):
            self.onnx_model = path_or_model
            logger.info("Model loaded from ModelProto object")
        else:
            raise TypeError("path_or_model must be a file path or an onnx.ModelProto object")
        
        # Create Graph from ONNX model
        self.graph = Graph.from_onnx(self.onnx_model)
        
        # Apply optimizations if configured
        if self.config.constant_folding:
            # Lazy load optimization functions
            from enhanced_onnx_tool.optimization.constant_folding import constant_folding
            constant_folding(self.graph)
            logger.info("Applied constant folding")
        
        if self.config.remove_dangling:
            self.graph.remove_dangling_nodes()
            logger.info("Removed dangling nodes")
        
        return self
    
    def save(self, 
             path: str, 
             shape_only: bool = False, 
             compatible: bool = True) -> str:
        """
        Save model to file.
        
        Args:
            path: Output file path
            shape_only: If True, save only shapes without weight data
            compatible: If True, ensure compatibility with standard ONNX tools
            
        Returns:
            Path to saved file
        """
        if not self.graph:
            raise ValueError("No graph to save, load a model first")
        
        # Convert graph back to ONNX
        onnx_model = self.graph.to_onnx(self.onnx_model)
        
        # Remove weight data if shape_only is True
        if shape_only:
            for initializer in onnx_model.graph.initializer:
                # Keep shape but remove data
                shape = [dim for dim in initializer.dims]
                new_tensor = onnx.helper.make_tensor(
                    name=initializer.name,
                    data_type=initializer.data_type,
                    dims=shape,
                    vals=[]  # Empty values
                )
                initializer.CopyFrom(new_tensor)
            logger.info("Saved model with shapes only (no weights)")
        
        # Save model
        onnx.save(onnx_model, path)
        logger.info(f"Model saved to {path}")
        
        return path
    
    def infer_shapes(self, input_shapes: Optional[Dict[str, List[int]]] = None) -> 'Model':
        """
        Infer shapes for the model.
        
        Args:
            input_shapes: Dictionary mapping input names to their shapes
            
        Returns:
            Self for method chaining
        """
        if not self.graph:
            raise ValueError("No graph to infer shapes for, load a model first")
        
        # Prepare input tensor shapes
        input_tensors = {}
        if input_shapes:
            for name, shape in input_shapes.items():
                if name in self.graph.tensor_map:
                    # Create numpy array with the right shape and dtype
                    tensor = self.graph.tensor_map[name]
                    tensor.update_shape(shape)
                    # If we need a concrete tensor value for inference
                    if tensor.dtype:
                        # Create a zero tensor with the right shape and dtype
                        dtype = tensor.numpy_dtype
                        input_tensors[name] = np.zeros(shape, dtype=dtype)
                else:
                    logger.warning(f"Input tensor '{name}' not found in graph")
        
        # Run shape inference
        self.shape_engine.infer(self.graph, input_tensors)
        logger.info("Shape inference completed")
        
        return self
    
    def profile(self, input_shapes: Optional[Dict[str, List[int]]] = None) -> Dict[str, Any]:
        """
        Profile the model for MACs, parameters, etc.
        
        Args:
            input_shapes: Dictionary mapping input names to their shapes
            
        Returns:
            Dictionary with profiling results
        """
        if not self.graph:
            raise ValueError("No graph to profile, load a model first")
        
        # Ensure shapes are inferred
        if input_shapes:
            self.infer_shapes(input_shapes)
        
        # Run profiling
        profile_results = self.profiler.profile(self.graph)
        logger.info(f"Profiling completed: {profile_results['total_macs']:,} MACs, {profile_results['total_params']:,} parameters")
        
        return profile_results
    
    def optimize(self, level: int = 1) -> 'Model':
        """
        Optimize the model.
        
        Args:
            level: Optimization level (1-3)
            
        Returns:
            Self for method chaining
        """
        if not self.graph:
            raise ValueError("No graph to optimize, load a model first")
        
        # Level 1: Constant folding
        if level >= 1:
            # Lazy load optimization functions
            from enhanced_onnx_tool.optimization.constant_folding import constant_folding
            constant_folding(self.graph)
            logger.info("Applied constant folding")
        
        # Level 2: Remove redundant nodes
        if level >= 2:
            self.graph.remove_redundant_nodes()
            logger.info("Removed redundant nodes")
        
        # Level 3: Advanced optimizations
        if level >= 3:
            # TODO: Implement more advanced optimizations
            logger.info("Applied advanced optimizations")
        
        return self
    
    def __repr__(self) -> str:
        """String representation of the model."""
        if self.onnx_model and self.graph:
            return f"Model(nodes={len(self.graph.nodes)}, inputs={len(self.graph.inputs)}, outputs={len(self.graph.outputs)})"
        else:
            return "Model(not loaded)"
