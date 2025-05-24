"""
Core type definitions for Enhanced ONNX Tool.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union

class DataType(Enum):
    """Enum of ONNX data types."""
    UNDEFINED = 0
    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15
    BFLOAT16 = 16

# 类型别名，简化引用
Shape = List[int]
TensorDict = Dict[str, 'Tensor']  # 前向引用
NodeDict = Dict[str, 'Node']      # 前向引用
