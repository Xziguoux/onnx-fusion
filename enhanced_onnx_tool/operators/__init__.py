"""
Operators module initialization.
"""

# Import registry first
from enhanced_onnx_tool.operators.registry import OPERATOR_REGISTRY

# Import all base operators to register them
from enhanced_onnx_tool.operators.base import (
    AddNode,
    MulNode,
    MatMulNode,
    ReshapeNode, 
    ConvNode,
    EqualNode,
    ShapeNode,
    UnsqueezeNode,
    WhereNode,
    GatherNode,
    ExpandNode,
    ConcatNode,
    TransposeNode,
    SliceNode,
    SoftmaxNode,
    PowNode,
    ReduceMeanNode,
    SqrtNode,
    DivNode,
    SigmoidNode
)
