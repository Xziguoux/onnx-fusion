"""
Tensor class for Enhanced ONNX Tool.
"""

import onnx
import numpy as np
from typing import List, Optional, Union

from enhanced_onnx_tool.core.types import DataType, Shape
from enhanced_onnx_tool.utils.logging import get_logger

logger = get_logger(__name__)

# Map from ONNX data types to numpy data types
ONNX_TO_NUMPY_DTYPE = {
    DataType.FLOAT.value: np.float32,
    DataType.UINT8.value: np.uint8,
    DataType.INT8.value: np.int8,
    DataType.UINT16.value: np.uint16,
    DataType.INT16.value: np.int16,
    DataType.INT32.value: np.int32,
    DataType.INT64.value: np.int64,
    DataType.BOOL.value: np.bool_,
    DataType.FLOAT16.value: np.float16,
    DataType.DOUBLE.value: np.float64,
    DataType.UINT32.value: np.uint32,
    DataType.UINT64.value: np.uint64,
    DataType.COMPLEX64.value: np.complex64,
    DataType.COMPLEX128.value: np.complex128,
    # BFLOAT16 doesn't have a direct numpy equivalent, use float32 as fallback
    DataType.BFLOAT16.value: np.float32,
}

# Map from numpy data types to ONNX data types
NUMPY_TO_ONNX_DTYPE = {
    np.dtype('float32'): DataType.FLOAT.value,
    np.dtype('uint8'): DataType.UINT8.value,
    np.dtype('int8'): DataType.INT8.value,
    np.dtype('uint16'): DataType.UINT16.value,
    np.dtype('int16'): DataType.INT16.value,
    np.dtype('int32'): DataType.INT32.value,
    np.dtype('int64'): DataType.INT64.value,
    np.dtype('bool'): DataType.BOOL.value,
    np.dtype('float16'): DataType.FLOAT16.value,
    np.dtype('float64'): DataType.DOUBLE.value,
    np.dtype('uint32'): DataType.UINT32.value,
    np.dtype('uint64'): DataType.UINT64.value,
    np.dtype('complex64'): DataType.COMPLEX64.value,
    np.dtype('complex128'): DataType.COMPLEX128.value,
}

class Tensor:
    """Represents a tensor in the computation graph."""
    
    def __init__(self, 
                 name: str = "", 
                 dtype: Optional[int] = None, 
                 shape: Optional[Shape] = None, 
                 data: Optional[np.ndarray] = None):
        """
        Initialize a tensor.
        
        Args:
            name: Tensor name
            dtype: ONNX data type (optional)
            shape: Tensor shape (optional)
            data: Tensor data as numpy array (optional)
        """
        self.name = name
        self.dtype = dtype
        self.shape = shape or []
        self._data = None
        self.is_constant = False
        
        # Set data if provided
        if data is not None:
            self.numpy = data
    
    @classmethod
    def from_onnx(cls, value_info: onnx.ValueInfoProto) -> 'Tensor':
        """
        Create a Tensor from an ONNX ValueInfoProto.
        
        Args:
            value_info: ONNX ValueInfoProto object
            
        Returns:
            New Tensor instance
        """
        tensor = cls(value_info.name)
        
        # Parse type and shape if available
        if value_info.type.HasField('tensor_type'):
            tensor_type = value_info.type.tensor_type
            
            # Set data type
            tensor.dtype = tensor_type.elem_type
            
            # Parse shape
            if tensor_type.HasField('shape'):
                shape = []
                for dim in tensor_type.shape.dim:
                    if dim.HasField('dim_value'):
                        shape.append(dim.dim_value)
                    else:
                        # Dynamic dimension, represent as -1
                        shape.append(-1)
                tensor.shape = shape
        
        return tensor
    
    @classmethod
    def from_onnx_tensor(cls, tensor: onnx.TensorProto) -> 'Tensor':
        """
        Create a Tensor from an ONNX TensorProto.
        
        Args:
            tensor: ONNX TensorProto object
            
        Returns:
            New Tensor instance
        """
        # Create tensor with name and data type
        result = cls(tensor.name, tensor.data_type)
        
        # Set shape
        result.shape = list(tensor.dims)
        
        # Convert raw data to numpy array
        np_array = onnx.numpy_helper.to_array(tensor)
        result.numpy = np_array
        
        return result
    
    def to_onnx_value_info(self) -> onnx.ValueInfoProto:
        """
        Convert the Tensor to an ONNX ValueInfoProto.
        
        Returns:
            ONNX ValueInfoProto
        """
        # Create type proto
        type_proto = onnx.TypeProto()
        tensor_type = type_proto.tensor_type
        
        # Set element type
        if self.dtype is not None:
            tensor_type.elem_type = self.dtype
        else:
            # Default to FLOAT if unknown
            tensor_type.elem_type = DataType.FLOAT.value
        
        # Set shape
        if self.shape:
            for dim in self.shape:
                tensor_type.shape.dim.add().dim_value = dim
        
        # Create value info
        value_info = onnx.ValueInfoProto()
        value_info.name = self.name
        value_info.type.CopyFrom(type_proto)
        
        return value_info
    
    def to_onnx_tensor(self) -> onnx.TensorProto:
        """
        Convert the Tensor to an ONNX TensorProto.
        
        Returns:
            ONNX TensorProto
        """
        if self._data is None:
            raise ValueError(f"Cannot convert tensor '{self.name}' to TensorProto: no data available")
        
        return onnx.numpy_helper.from_array(self._data, name=self.name)
    
    @property
    def numpy(self) -> Optional[np.ndarray]:
        """Get tensor data as numpy array."""
        return self._data
    
    @numpy.setter
    def numpy(self, array: np.ndarray):
        """Set tensor data from numpy array."""
        if array is None:
            self._data = None
            self.is_constant = False
            return
        
        # Store array
        self._data = array
        self.is_constant = True
        
        # Update shape and dtype based on array
        self.shape = list(array.shape)
        
        # Set dtype if not already set
        if self.dtype is None:
            np_dtype = array.dtype
            if np_dtype in NUMPY_TO_ONNX_DTYPE:
                self.dtype = NUMPY_TO_ONNX_DTYPE[np_dtype]
            else:
                # Default to FLOAT if unknown
                logger.warning(f"Unknown numpy dtype {np_dtype}, defaulting to FLOAT")
                self.dtype = DataType.FLOAT.value
    
    @property
    def numpy_dtype(self) -> np.dtype:
        """Get numpy data type for this tensor."""
        if self.dtype is None:
            # Default to float32 if unknown
            return np.dtype('float32')
        
        if self.dtype in ONNX_TO_NUMPY_DTYPE:
            return np.dtype(ONNX_TO_NUMPY_DTYPE[self.dtype])
        else:
            logger.warning(f"Unknown ONNX dtype {self.dtype}, defaulting to float32")
            return np.dtype('float32')
    
    def update_shape(self, shape: Shape) -> None:
        """
        Update tensor shape.
        
        Args:
            shape: New shape
        """
        self.shape = shape
        
        # Update data shape if constant tensor
        if self.is_constant and self._data is not None:
            # Check if total size matches
            old_size = self._data.size
            new_size = np.prod(shape)
            
            if old_size == new_size:
                # Reshape data to match new shape
                self._data = self._data.reshape(shape)
            else:
                logger.warning(
                    f"Cannot reshape tensor '{self.name}' data from shape {self._data.shape} to {shape}: "
                    f"size mismatch ({old_size} vs {new_size})"
                )
    
    def is_valid_shape(self) -> bool:
        """
        Check if tensor has a valid shape.
        
        Returns:
            True if shape is valid
        """
        return self.shape is not None and all(isinstance(dim, int) and dim > 0 for dim in self.shape)
    
    def __repr__(self) -> str:
        """String representation of the tensor."""
        dtype_str = f"dtype={self.dtype}" if self.dtype is not None else "dtype=unknown"
        shape_str = f"shape={self.shape}" if self.shape else "shape=unknown"
        constant_str = "constant" if self.is_constant else "variable"
        return f"Tensor(name='{self.name}', {dtype_str}, {shape_str}, {constant_str})"
