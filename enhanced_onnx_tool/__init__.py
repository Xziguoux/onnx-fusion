"""
Enhanced ONNX Tool - A robust parser, editor and profiler for ONNX models.
"""

# Import operators first to register them
from enhanced_onnx_tool.operators import OPERATOR_REGISTRY

# Import version information
from enhanced_onnx_tool.version import __version__

# Import core classes
from enhanced_onnx_tool.model import Model, ModelConfig
from enhanced_onnx_tool.graph.graph import Graph
from enhanced_onnx_tool.core.node import Node
from enhanced_onnx_tool.core.tensor import Tensor, DataType

# 辅助函数定义
def load_model(path_or_model, **kwargs):
    """Load an ONNX model."""
    return Model(path_or_model, **kwargs)

def infer_shapes(model_path, input_shapes=None, output_path=None):
    """Infer shapes for an ONNX model."""
    model = Model(model_path)
    model.infer_shapes(input_shapes)
    if output_path:
        model.save(output_path)
    return model

def profile_model(model_path, input_shapes=None):
    """Profile an ONNX model."""
    model = Model(model_path)
    return model.profile(input_shapes)

def optimize_model(model_path, output_path=None, level=1):
    """Optimize an ONNX model."""
    model = Model(model_path)
    model.optimize(level)
    if output_path:
        model.save(output_path)
    return model
