"""
Base operators for Enhanced ONNX Tool.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from enhanced_onnx_tool.core.node import Node
from enhanced_onnx_tool.core.tensor import Tensor
from enhanced_onnx_tool.operators.registry import OPERATOR_REGISTRY
from enhanced_onnx_tool.utils.logging import get_logger

logger = get_logger(__name__)

# 添加一个通用的GenericNode类，用于处理未注册的算子
class GenericNode(Node):
    """
    通用算子节点，用于处理未注册的ONNX算子。
    将简单地传递输入形状，不做任何实际的形状推断。
    """
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        对于未知算子，简单地将输入形状作为输出形状返回。
        这是一个保守的做法，不会导致错误，但可能不准确。
        
        Args:
            input_tensors: 输入张量列表
            
        Returns:
            输出形状列表
        """
        logger.warning(f"Using generic shape inference for operator: {self.op_type}")
        
        # 如果没有输入，返回None
        if not input_tensors:
            return [None]
        
        # 默认情况下，假设输出形状与第一个输入相同
        # 这是一个保守的假设，适用于许多元素级操作
        return [input_tensors[0].shape if input_tensors else None]
    
    def value_infer(self, input_tensors: List[Tensor]) -> List[np.ndarray]:
        """
        对于未知算子，返回None值。
        
        Args:
            input_tensors: 输入张量列表
            
        Returns:
            None
        """
        logger.warning(f"Value inference not implemented for operator: {self.op_type}")
        return [None]
    
    def profile(self, input_tensors: List[Tensor], output_tensors: List[Tensor]) -> Tuple[int, int]:
        """
        对于未知算子，返回0 MACs和0参数。
        
        Args:
            input_tensors: 输入张量列表
            output_tensors: 输出张量列表
            
        Returns:
            (0, 0)表示未知计算量
        """
        logger.warning(f"Profiling not implemented for operator: {self.op_type}")
        return 0, 0

@OPERATOR_REGISTRY.register("Cast")
class CastNode(Node):
    """Cast算子的实现。"""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        推断Cast算子的输出形状。
        Cast不改变形状，仅改变数据类型。
        
        Args:
            input_tensors: 输入张量列表
            
        Returns:
            输出形状列表
        """
        if len(input_tensors) != 1:
            raise ValueError(f"Cast expects 1 input, got {len(input_tensors)}")
        
        data = input_tensors[0]
        
        if not data.shape:
            raise ValueError("Cast input must have a shape")
        
        # Cast不改变形状
        return [data.shape]
    
    def value_infer(self, input_tensors: List[Tensor]) -> List[np.ndarray]:
        """
        计算Cast算子的输出值。
        
        Args:
            input_tensors: 输入张量列表
            
        Returns:
            输出numpy数组列表
        """
        if len(input_tensors) != 1:
            raise ValueError(f"Cast expects 1 input, got {len(input_tensors)}")
        
        data = input_tensors[0].numpy
        
        if data is None:
            raise ValueError("Cast input must have a value")
        
        # 获取目标类型
        to_type = self.get_attr("to", None)
        if to_type is None:
            raise ValueError("Cast requires 'to' attribute")
        
        # ONNX类型到numpy类型的映射
        onnx_to_numpy = {
            1: np.float32,   # FLOAT
            2: np.uint8,     # UINT8
            3: np.int8,      # INT8
            4: np.uint16,    # UINT16
            5: np.int16,     # INT16
            6: np.int32,     # INT32
            7: np.int64,     # INT64
            9: np.bool_,     # BOOL
            10: np.float16,  # FLOAT16
            11: np.double,   # DOUBLE
            12: np.uint32,   # UINT32
            13: np.uint64,   # UINT64
            14: np.complex64,# COMPLEX64
            15: np.complex128 # COMPLEX128
        }
        
        if to_type not in onnx_to_numpy:
            logger.warning(f"Unsupported Cast target type: {to_type}")
            return [data]
        
        try:
            return [data.astype(onnx_to_numpy[to_type])]
        except Exception as e:
            logger.error(f"Cast failed: {str(e)}")
            return [data]

@OPERATOR_REGISTRY.register("Add")
class AddNode(Node):
    """Implementation of the Add operator."""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        Infer output shapes for Add operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        if len(input_tensors) != 2:
            raise ValueError(f"Add expects 2 inputs, got {len(input_tensors)}")
        
        shape_a = input_tensors[0].shape
        shape_b = input_tensors[1].shape
        
        if not shape_a or not shape_b:
            raise ValueError(f"Add inputs must have shapes: {shape_a}, {shape_b}")
        
        # Implement broadcasting rules
        return [self._broadcast_shapes(shape_a, shape_b)]
    
    def value_infer(self, input_tensors: List[Tensor]) -> List[np.ndarray]:
        """
        Compute output values for Add operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output numpy arrays
        """
        if len(input_tensors) != 2:
            raise ValueError(f"Add expects 2 inputs, got {len(input_tensors)}")
        
        a = input_tensors[0].numpy
        b = input_tensors[1].numpy
        
        if a is None or b is None:
            raise ValueError("Add inputs must have values")
        
        return [np.add(a, b)]
    
    def profile(self, input_tensors: List[Tensor], output_tensors: List[Tensor]) -> Tuple[int, int]:
        """
        Profile the Add operator.
        
        Args:
            input_tensors: List of input tensors
            output_tensors: List of output tensors
            
        Returns:
            Tuple of (MACs, parameters)
        """
        # Add has no learnable parameters and uses one operation per element
        output_size = np.prod(output_tensors[0].shape) if output_tensors[0].shape else 0
        return output_size, 0
    
    def _broadcast_shapes(self, shape_a: List[int], shape_b: List[int]) -> List[int]:
        """
        Apply numpy broadcasting rules to shapes.
        
        Args:
            shape_a: First shape
            shape_b: Second shape
            
        Returns:
            Broadcasted shape
        """
        # Convert to numpy arrays for easier calculation
        a = np.zeros(shape_a)
        b = np.zeros(shape_b)
        
        # Use numpy broadcasting rules
        c = a + b
        
        return list(c.shape)

@OPERATOR_REGISTRY.register("Mul")
class MulNode(Node):
    """Implementation of the Mul operator."""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        Infer output shapes for Mul operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        if len(input_tensors) != 2:
            raise ValueError(f"Mul expects 2 inputs, got {len(input_tensors)}")
        
        shape_a = input_tensors[0].shape
        shape_b = input_tensors[1].shape
        
        if not shape_a or not shape_b:
            raise ValueError(f"Mul inputs must have shapes: {shape_a}, {shape_b}")
        
        # Implement broadcasting rules (same as Add)
        return [self._broadcast_shapes(shape_a, shape_b)]
    
    def value_infer(self, input_tensors: List[Tensor]) -> List[np.ndarray]:
        """
        Compute output values for Mul operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output numpy arrays
        """
        if len(input_tensors) != 2:
            raise ValueError(f"Mul expects 2 inputs, got {len(input_tensors)}")
        
        a = input_tensors[0].numpy
        b = input_tensors[1].numpy
        
        if a is None or b is None:
            raise ValueError("Mul inputs must have values")
        
        return [np.multiply(a, b)]
    
    def profile(self, input_tensors: List[Tensor], output_tensors: List[Tensor]) -> Tuple[int, int]:
        """
        Profile the Mul operator.
        
        Args:
            input_tensors: List of input tensors
            output_tensors: List of output tensors
            
        Returns:
            Tuple of (MACs, parameters)
        """
        # Mul has no learnable parameters and uses one operation per element
        output_size = np.prod(output_tensors[0].shape) if output_tensors[0].shape else 0
        return output_size, 0
    
    def _broadcast_shapes(self, shape_a: List[int], shape_b: List[int]) -> List[int]:
        """
        Apply numpy broadcasting rules to shapes.
        
        Args:
            shape_a: First shape
            shape_b: Second shape
            
        Returns:
            Broadcasted shape
        """
        # Convert to numpy arrays for easier calculation
        a = np.zeros(shape_a)
        b = np.zeros(shape_b)
        
        # Use numpy broadcasting rules
        c = a * b
        
        return list(c.shape)

@OPERATOR_REGISTRY.register("MatMul")
class MatMulNode(Node):
    """Implementation of the MatMul operator."""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        Infer output shapes for MatMul operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        if len(input_tensors) != 2:
            raise ValueError(f"MatMul expects 2 inputs, got {len(input_tensors)}")
        
        shape_a = input_tensors[0].shape
        shape_b = input_tensors[1].shape
        
        if not shape_a or not shape_b:
            raise ValueError(f"MatMul inputs must have shapes: {shape_a}, {shape_b}")
        
        # Simple 2D matrix multiplication
        if len(shape_a) == 2 and len(shape_b) == 2:
            if shape_a[1] != shape_b[0]:
                raise ValueError(f"Incompatible shapes for MatMul: {shape_a} and {shape_b}")
            return [[shape_a[0], shape_b[1]]]
        
        # Batched matrix multiplication
        return [self._batched_matmul_shape(shape_a, shape_b)]
    
    def value_infer(self, input_tensors: List[Tensor]) -> List[np.ndarray]:
        """
        Compute output values for MatMul operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output numpy arrays
        """
        if len(input_tensors) != 2:
            raise ValueError(f"MatMul expects 2 inputs, got {len(input_tensors)}")
        
        a = input_tensors[0].numpy
        b = input_tensors[1].numpy
        
        if a is None or b is None:
            raise ValueError("MatMul inputs must have values")
        
        return [np.matmul(a, b)]
    
    def profile(self, input_tensors: List[Tensor], output_tensors: List[Tensor]) -> Tuple[int, int]:
        """
        Profile the MatMul operator.
        
        Args:
            input_tensors: List of input tensors
            output_tensors: List of output tensors
            
        Returns:
            Tuple of (MACs, parameters)
        """
        shape_a = input_tensors[0].shape
        shape_b = input_tensors[1].shape
        
        if not shape_a or not shape_b:
            return 0, 0
        
        # Simple 2D matrix multiplication
        if len(shape_a) == 2 and len(shape_b) == 2:
            # M*N*K multiply-adds
            M, K = shape_a
            K2, N = shape_b
            
            if K != K2:
                return 0, 0
                
            # Each output element requires K multiply-adds
            return M * N * K, 0
        
        # Batched matrix multiplication is more complex
        # For simplicity, compute the total number of multiply-adds
        output_shape = output_tensors[0].shape
        if not output_shape:
            return 0, 0
        
        # For batched matmul, find the inner dimension (K) from inputs
        K = shape_a[-1]
        
        # Total number of elements in the output
        output_size = np.prod(output_shape)
        
        # Each output element requires K multiply-adds
        return output_size * K, 0
    
    def _batched_matmul_shape(self, shape_a: List[int], shape_b: List[int]) -> List[int]:
        """
        Calculate output shape for batched matrix multiplication.
        
        Args:
            shape_a: First shape
            shape_b: Second shape
            
        Returns:
            Output shape
        """
        # Ensure at least 2D
        if len(shape_a) < 2 or len(shape_b) < 2:
            raise ValueError(f"MatMul inputs must be at least 2D: {shape_a} and {shape_b}")
        
        # Check inner dimensions match
        if shape_a[-1] != shape_b[-2]:
            raise ValueError(f"Incompatible inner dimensions for MatMul: {shape_a} and {shape_b}")
        
        # Get batch dimensions through broadcasting
        batch_a = shape_a[:-2]
        batch_b = shape_b[:-2]
        
        # Broadcast batch dimensions
        if batch_a and batch_b:
            try:
                batch_out = self._broadcast_shapes(batch_a, batch_b)
            except ValueError:
                raise ValueError(f"Incompatible batch dimensions for MatMul: {batch_a} and {batch_b}")
        else:
            # If either input has no batch dims, use the other's batch dims
            batch_out = batch_a or batch_b
        
        # Output shape is [*batch_out, shape_a[-2], shape_b[-1]]
        return batch_out + [shape_a[-2], shape_b[-1]]

@OPERATOR_REGISTRY.register("Reshape")
class ReshapeNode(Node):
    """Implementation of the Reshape operator."""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        Infer output shapes for Reshape operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        if len(input_tensors) != 2:
            raise ValueError(f"Reshape expects 2 inputs, got {len(input_tensors)}")
        
        data = input_tensors[0]
        shape_tensor = input_tensors[1]
        
        if not data.shape:
            raise ValueError("First input to Reshape must have a valid shape")
        
        # If shape_tensor is constant, we can determine the output shape
        if shape_tensor.is_constant and shape_tensor.numpy is not None:
            shape = shape_tensor.numpy.tolist()
            
            # Handle dynamic dimensions (denoted by -1)
            if -1 in shape:
                # Calculate the total size of the tensor
                data_size = np.prod(data.shape)
                
                # Calculate the size of the known dimensions
                known_size = 1
                for dim in shape:
                    if dim != -1:
                        known_size *= dim
                
                # Calculate the size of the unknown dimension
                if data_size % known_size != 0:
                    raise ValueError(f"Cannot reshape tensor of size {data_size} to shape {shape}")
                
                unknown_size = data_size // known_size
                
                # Replace -1 with the calculated size
                shape = [unknown_size if dim == -1 else dim for dim in shape]
            
            return [shape]
        
        # If shape is not constant, we can't determine the output shape
        logger.warning("Cannot infer output shape for Reshape: shape tensor is not constant")
        return [None]
    
    def value_infer(self, input_tensors: List[Tensor]) -> List[np.ndarray]:
        """
        Compute output values for Reshape operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output numpy arrays
        """
        if len(input_tensors) != 2:
            raise ValueError(f"Reshape expects 2 inputs, got {len(input_tensors)}")
        
        data = input_tensors[0].numpy
        shape_tensor = input_tensors[1].numpy
        
        if data is None or shape_tensor is None:
            raise ValueError("Reshape inputs must have values")
        
        # Get the new shape
        new_shape = shape_tensor.tolist()
        
        # Apply reshape
        try:
            reshaped = data.reshape(new_shape)
            return [reshaped]
        except Exception as e:
            raise ValueError(f"Failed to reshape data: {str(e)}")
    
    def profile(self, input_tensors: List[Tensor], output_tensors: List[Tensor]) -> Tuple[int, int]:
        """
        Profile the Reshape operator.
        
        Args:
            input_tensors: List of input tensors
            output_tensors: List of output tensors
            
        Returns:
            Tuple of (MACs, parameters)
        """
        # Reshape is a data movement operation, no compute
        return 0, 0

@OPERATOR_REGISTRY.register("Conv")
class ConvNode(Node):
    """Implementation of the Conv operator."""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        Infer output shapes for Conv operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        if len(input_tensors) < 2:
            raise ValueError(f"Conv expects at least 2 inputs, got {len(input_tensors)}")
        
        x = input_tensors[0]
        w = input_tensors[1]
        
        if not x.shape or not w.shape:
            raise ValueError("Conv inputs must have shapes")
        
        # Get attributes
        auto_pad = self.get_attr("auto_pad", "NOTSET")
        dilations = self.get_attr("dilations", [1] * (len(x.shape) - 2))
        group = self.get_attr("group", 1)
        kernel_shape = self.get_attr("kernel_shape", w.shape[2:])
        pads = self.get_attr("pads", [0] * (len(x.shape) - 2) * 2)
        strides = self.get_attr("strides", [1] * (len(x.shape) - 2))
        
        # Validate inputs
        if len(x.shape) < 3:
            raise ValueError(f"Conv input X must be at least 3D, got {len(x.shape)}D")
        
        if len(w.shape) != len(x.shape):
            raise ValueError(f"Conv input W must have the same dimensionality as X, got {len(w.shape)} vs {len(x.shape)}")
        
        # Compute output shape
        N = x.shape[0]  # Batch size
        M = w.shape[0]  # Output channels
        
        # Spatial dimensions
        spatial_shape = []
        for i in range(len(x.shape) - 2):
            dim = x.shape[i + 2]
            kernel = kernel_shape[i]
            stride = strides[i]
            dilation = dilations[i]
            pad_begin = pads[i]
            pad_end = pads[i + len(x.shape) - 2]
            
            effective_kernel = (kernel - 1) * dilation + 1
            
            if auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
                output_dim = (dim + stride - 1) // stride
                pad_needed = max(0, (output_dim - 1) * stride + effective_kernel - dim)
                if auto_pad == "SAME_UPPER":
                    pad_begin = pad_needed // 2
                    pad_end = pad_needed - pad_begin
                else:
                    pad_end = pad_needed // 2
                    pad_begin = pad_needed - pad_end
            else:
                output_dim = (dim + pad_begin + pad_end - effective_kernel) // stride + 1
            
            spatial_shape.append(output_dim)
        
        return [[N, M] + spatial_shape]
    
    def value_infer(self, input_tensors: List[Tensor]) -> List[np.ndarray]:
        """
        Compute output values for Conv operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output numpy arrays
        """
        # Value inference for Conv is complex and not always necessary
        # For constant folding, we might implement this later
        logger.warning("Value inference for Conv is not implemented")
        return [None]
    
    def profile(self, input_tensors: List[Tensor], output_tensors: List[Tensor]) -> Tuple[int, int]:
        """
        Profile the Conv operator.
        
        Args:
            input_tensors: List of input tensors
            output_tensors: List of output tensors
            
        Returns:
            Tuple of (MACs, parameters)
        """
        if len(input_tensors) < 2 or not output_tensors:
            return 0, 0
        
        x = input_tensors[0]
        w = input_tensors[1]
        
        if not x.shape or not w.shape or not output_tensors[0].shape:
            return 0, 0
        
        # Get filter dimensions
        out_channels = w.shape[0]
        in_channels = w.shape[1]
        filter_shape = w.shape[2:]
        
        # Get attributes
        group = self.get_attr("group", 1)
        
        # Number of parameters
        params = np.prod(w.shape)
        
        # Add bias parameters if present
        if len(input_tensors) > 2 and input_tensors[2].shape:
            params += np.prod(input_tensors[2].shape)
        
        # Compute MACs: one MAC per filter element per output element
        filter_elements = np.prod(filter_shape) * (in_channels / group)
        output_elements = np.prod(output_tensors[0].shape)
        
        # Each output element requires filter_elements * in_channels multiply-adds
        macs = output_elements * filter_elements
        
        return int(macs), int(params)

@OPERATOR_REGISTRY.register("Equal")
class EqualNode(Node):
    """Implementation of the Equal operator."""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        Infer output shapes for Equal operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        if len(input_tensors) != 2:
            raise ValueError(f"Equal expects 2 inputs, got {len(input_tensors)}")
        
        shape_a = input_tensors[0].shape
        shape_b = input_tensors[1].shape
        
        if not shape_a or not shape_b:
            raise ValueError(f"Equal inputs must have shapes: {shape_a}, {shape_b}")
        
        # Implement broadcasting rules
        return [self._broadcast_shapes(shape_a, shape_b)]
    
    def _broadcast_shapes(self, shape_a: List[int], shape_b: List[int]) -> List[int]:
        """Apply numpy broadcasting rules to shapes."""
        # Convert to numpy arrays for easier calculation
        a = np.zeros(shape_a)
        b = np.zeros(shape_b)
        
        # Use numpy broadcasting rules
        try:
            c = np.broadcast_arrays(a, b)[0]
            return list(c.shape)
        except ValueError:
            raise ValueError(f"Cannot broadcast shapes {shape_a} and {shape_b}")

@OPERATOR_REGISTRY.register("Shape")
class ShapeNode(Node):
    """Implementation of the Shape operator."""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        Infer output shapes for Shape operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        if len(input_tensors) != 1:
            raise ValueError(f"Shape expects 1 input, got {len(input_tensors)}")
        
        input_shape = input_tensors[0].shape
        
        if not input_shape:
            raise ValueError(f"Shape input must have a shape")
        
        # Output is a 1D tensor with length equal to the rank of the input
        return [[len(input_shape)]]
    
    def value_infer(self, input_tensors: List[Tensor]) -> List[np.ndarray]:
        """
        Compute output values for Shape operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output numpy arrays
        """
        if len(input_tensors) != 1:
            raise ValueError(f"Shape expects 1 input, got {len(input_tensors)}")
        
        shape = input_tensors[0].shape
        
        if not shape:
            raise ValueError(f"Shape input must have a shape")
        
        return [np.array(shape, dtype=np.int64)]

@OPERATOR_REGISTRY.register("Unsqueeze")
class UnsqueezeNode(Node):
    """Implementation of the Unsqueeze operator."""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        Infer output shapes for Unsqueeze operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        if len(input_tensors) not in (1, 2):
            raise ValueError(f"Unsqueeze expects 1 or 2 inputs, got {len(input_tensors)}")
        
        data = input_tensors[0]
        
        if not data.shape:
            raise ValueError(f"Unsqueeze input must have a shape")
        
        # Get axes from attribute or second input
        axes = None
        if len(input_tensors) == 2 and input_tensors[1].is_constant and input_tensors[1].numpy is not None:
            # Axes provided as second input
            axes = input_tensors[1].numpy.tolist()
        else:
            # Axes provided as attribute
            axes = self.get_attr("axes", [0])
        
        if axes is None:
            raise ValueError("Unsqueeze requires axes to be specified")
        
        # Convert negative axes to positive
        data_rank = len(data.shape)
        axes = [axis if axis >= 0 else axis + data_rank + 1 for axis in axes]
        
        # Insert dimensions of size 1 at the specified axes
        new_shape = list(data.shape)
        for axis in sorted(axes):
            new_shape.insert(axis, 1)
        
        return [new_shape]
    
    def value_infer(self, input_tensors: List[Tensor]) -> List[np.ndarray]:
        """
        Compute output values for Unsqueeze operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output numpy arrays
        """
        if len(input_tensors) not in (1, 2):
            raise ValueError(f"Unsqueeze expects 1 or 2 inputs, got {len(input_tensors)}")
        
        data = input_tensors[0].numpy
        
        if data is None:
            raise ValueError("Unsqueeze input must have a value")
        
        # Get axes from attribute or second input
        axes = None
        if len(input_tensors) == 2 and input_tensors[1].is_constant and input_tensors[1].numpy is not None:
            # Axes provided as second input
            axes = input_tensors[1].numpy.tolist()
        else:
            # Axes provided as attribute
            axes = self.get_attr("axes", [0])
        
        if axes is None:
            raise ValueError("Unsqueeze requires axes to be specified")
        
        # Convert negative axes to positive
        data_rank = len(data.shape)
        axes = [axis if axis >= 0 else axis + data_rank + 1 for axis in axes]
        
        # Insert dimensions of size 1 at the specified axes
        new_shape = list(data.shape)
        for axis in sorted(axes):
            new_shape.insert(axis, 1)
        
        return [data.reshape(new_shape)]

@OPERATOR_REGISTRY.register("Where")
class WhereNode(Node):
    """Implementation of the Where operator."""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        Infer output shapes for Where operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        if len(input_tensors) != 3:
            raise ValueError(f"Where expects 3 inputs, got {len(input_tensors)}")
        
        condition = input_tensors[0]
        x = input_tensors[1]
        y = input_tensors[2]
        
        if not condition.shape or not x.shape or not y.shape:
            raise ValueError(f"Where inputs must have shapes: {condition.shape}, {x.shape}, {y.shape}")
        
        # Broadcast all three shapes
        try:
            # Broadcast x and y first
            xy_shape = self._broadcast_shapes(x.shape, y.shape)
            
            # Then broadcast with condition
            return [self._broadcast_shapes(condition.shape, xy_shape)]
        except ValueError as e:
            raise ValueError(f"Cannot broadcast shapes for Where: {str(e)}")
    
    def _broadcast_shapes(self, shape_a: List[int], shape_b: List[int]) -> List[int]:
        """Apply numpy broadcasting rules to shapes."""
        # Convert to numpy arrays for easier calculation
        a = np.zeros(shape_a)
        b = np.zeros(shape_b)
        
        # Use numpy broadcasting rules
        try:
            c = np.broadcast_arrays(a, b)[0]
            return list(c.shape)
        except ValueError:
            raise ValueError(f"Cannot broadcast shapes {shape_a} and {shape_b}")

@OPERATOR_REGISTRY.register("Gather")
class GatherNode(Node):
    """Implementation of the Gather operator."""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        Infer output shapes for Gather operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        if len(input_tensors) != 2:
            raise ValueError(f"Gather expects 2 inputs, got {len(input_tensors)}")
        
        data = input_tensors[0]
        indices = input_tensors[1]
        
        if not data.shape or not indices.shape:
            raise ValueError(f"Gather inputs must have shapes: {data.shape}, {indices.shape}")
        
        # Get axis from attribute
        axis = self.get_attr("axis", 0)
        
        # Convert negative axis to positive
        data_rank = len(data.shape)
        axis = axis if axis >= 0 else axis + data_rank
        
        if axis < 0 or axis >= data_rank:
            raise ValueError(f"Gather axis {axis} is out of bounds for data rank {data_rank}")
        
        # Output shape is data.shape[:axis] + indices.shape + data.shape[axis+1:]
        output_shape = data.shape[:axis] + indices.shape + data.shape[axis+1:]
        
        return [output_shape]

@OPERATOR_REGISTRY.register("Expand")
class ExpandNode(Node):
    """Implementation of the Expand operator."""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        Infer output shapes for Expand operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        if len(input_tensors) != 2:
            raise ValueError(f"Expand expects 2 inputs, got {len(input_tensors)}")
        
        data = input_tensors[0]
        shape = input_tensors[1]
        
        if not data.shape:
            raise ValueError(f"Expand data input must have a shape")
        
        # If shape is constant, we can determine the output shape
        if shape.is_constant and shape.numpy is not None:
            new_shape = shape.numpy.tolist()
            
            # Apply broadcasting rules
            data_shape = data.shape
            data_rank = len(data_shape)
            new_rank = len(new_shape)
            
            # If the new shape has more dimensions, pad data shape with 1s
            if new_rank > data_rank:
                data_shape = [1] * (new_rank - data_rank) + data_shape
            
            # Apply broadcasting rules dimension by dimension
            output_shape = []
            for i in range(len(new_shape)):
                if i < len(data_shape):
                    if new_shape[i] == -1:
                        # -1 means use the original dimension
                        output_shape.append(data_shape[i])
                    elif new_shape[i] == 0:
                        # 0 means copy from the input
                        output_shape.append(data_shape[i])
                    else:
                        # Otherwise use the new dimension
                        output_shape.append(new_shape[i])
                else:
                    output_shape.append(new_shape[i])
            
            return [output_shape]
        
        # If shape is not constant, we can't determine the output shape
        logger.warning("Cannot infer output shape for Expand: shape is not constant")
        return [None]

@OPERATOR_REGISTRY.register("Concat")
class ConcatNode(Node):
    """Implementation of the Concat operator."""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        Infer output shapes for Concat operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        if len(input_tensors) < 1:
            raise ValueError(f"Concat expects at least 1 input, got {len(input_tensors)}")
        
        # Get axis from attribute
        axis = self.get_attr("axis", 0)
        
        # Check if all input tensors have shapes
        for i, tensor in enumerate(input_tensors):
            if not tensor.shape:
                raise ValueError(f"Concat input {i} must have a shape")
        
        # Get rank of first input
        first_rank = len(input_tensors[0].shape)
        
        # Convert negative axis to positive
        axis = axis if axis >= 0 else axis + first_rank
        
        if axis < 0 or axis >= first_rank:
            raise ValueError(f"Concat axis {axis} is out of bounds for input rank {first_rank}")
        
        # Check if all inputs have the same rank
        for i, tensor in enumerate(input_tensors):
            if len(tensor.shape) != first_rank:
                raise ValueError(f"Concat input {i} has rank {len(tensor.shape)}, but first input has rank {first_rank}")
        
        # Check if all inputs have the same shape except on the concat axis
        for i, tensor in enumerate(input_tensors[1:], 1):
            for j in range(first_rank):
                if j != axis and tensor.shape[j] != input_tensors[0].shape[j]:
                    raise ValueError(f"Concat input {i} has shape {tensor.shape}, but first input has shape {input_tensors[0].shape}")
        
        # Compute output shape
        output_shape = list(input_tensors[0].shape)
        for tensor in input_tensors[1:]:
            output_shape[axis] += tensor.shape[axis]
        
        return [output_shape]

@OPERATOR_REGISTRY.register("Transpose")
class TransposeNode(Node):
    """Implementation of the Transpose operator."""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        Infer output shapes for Transpose operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        if len(input_tensors) != 1:
            raise ValueError(f"Transpose expects 1 input, got {len(input_tensors)}")
        
        data = input_tensors[0]
        
        if not data.shape:
            raise ValueError(f"Transpose input must have a shape")
        
        # Get perm from attribute
        perm = self.get_attr("perm", None)
        
        if perm is None:
            # Default is to reverse the dimensions
            perm = list(range(len(data.shape)))
            perm.reverse()
        
        # Check if perm is valid
        if len(perm) != len(data.shape):
            raise ValueError(f"Transpose perm {perm} has length {len(perm)}, but input has rank {len(data.shape)}")
        
        # Compute output shape
        output_shape = [data.shape[i] for i in perm]
        
        return [output_shape]
    
    def value_infer(self, input_tensors: List[Tensor]) -> List[np.ndarray]:
        """
        Compute output values for Transpose operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output numpy arrays
        """
        if len(input_tensors) != 1:
            raise ValueError(f"Transpose expects 1 input, got {len(input_tensors)}")
        
        data = input_tensors[0].numpy
        
        if data is None:
            raise ValueError("Transpose input must have a value")
        
        # Get perm from attribute
        perm = self.get_attr("perm", None)
        
        if perm is None:
            # Default is to reverse the dimensions
            perm = list(range(len(data.shape)))
            perm.reverse()
        
        # Check if perm is valid
        if len(perm) != len(data.shape):
            raise ValueError(f"Transpose perm {perm} has length {len(perm)}, but input has rank {len(data.shape)}")
        
        return [np.transpose(data, perm)]

@OPERATOR_REGISTRY.register("Slice")
class SliceNode(Node):
    """Implementation of the Slice operator."""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        Infer output shapes for Slice operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        if len(input_tensors) < 1:
            raise ValueError(f"Slice expects at least 1 input, got {len(input_tensors)}")
        
        data = input_tensors[0]
        
        if not data.shape:
            raise ValueError(f"Slice input must have a shape")
        
        # Get starts, ends, axes, steps from attributes or inputs
        starts = None
        ends = None
        axes = None
        steps = None
        
        # ONNX Slice has two versions: one with attributes and one with inputs
        if len(input_tensors) >= 3:
            # Slice-10 (inputs version)
            if input_tensors[1].is_constant and input_tensors[1].numpy is not None:
                starts = input_tensors[1].numpy.tolist()
            if input_tensors[2].is_constant and input_tensors[2].numpy is not None:
                ends = input_tensors[2].numpy.tolist()
            if len(input_tensors) >= 4 and input_tensors[3].is_constant and input_tensors[3].numpy is not None:
                axes = input_tensors[3].numpy.tolist()
            if len(input_tensors) >= 5 and input_tensors[4].is_constant and input_tensors[4].numpy is not None:
                steps = input_tensors[4].numpy.tolist()
        else:
            # Slice-1 (attributes version)
            starts = self.get_attr("starts", None)
            ends = self.get_attr("ends", None)
            axes = self.get_attr("axes", None)
            steps = self.get_attr("steps", None)
        
        if starts is None or ends is None:
            logger.warning("Cannot infer output shape for Slice: starts or ends is not available")
            return [None]
        
        # Default axes is [0, 1, ...]
        if axes is None:
            axes = list(range(len(starts)))
        
        # Default steps is [1, 1, ...]
        if steps is None:
            steps = [1] * len(starts)
        
        # Check if all arrays have the same length
        if not (len(starts) == len(ends) == len(axes) == len(steps)):
            raise ValueError(f"Slice parameters must have the same length: starts={len(starts)}, ends={len(ends)}, axes={len(axes)}, steps={len(steps)}")
        
        # Convert negative axes to positive
        data_rank = len(data.shape)
        axes = [axis if axis >= 0 else axis + data_rank for axis in axes]
        
        # Compute output shape
        output_shape = list(data.shape)
        for i, axis in enumerate(axes):
            start = starts[i]
            end = ends[i]
            step = steps[i]
            
            # Handle negative indices
            if start < 0:
                start += data.shape[axis]
            if end < 0:
                end += data.shape[axis]
            
            # Clamp to valid range
            start = max(0, min(start, data.shape[axis]))
            end = max(0, min(end, data.shape[axis]))
            
            # Compute sliced dimension size
            if step > 0:
                output_shape[axis] = (end - start + step - 1) // step
            else:
                output_shape[axis] = (start - end - step - 1) // (-step)
        
        return [output_shape]

@OPERATOR_REGISTRY.register("Softmax")
class SoftmaxNode(Node):
    """Implementation of the Softmax operator."""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        Infer output shapes for Softmax operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        if len(input_tensors) != 1:
            raise ValueError(f"Softmax expects 1 input, got {len(input_tensors)}")
        
        data = input_tensors[0]
        
        if not data.shape:
            raise ValueError(f"Softmax input must have a shape")
        
        # Softmax doesn't change the shape
        return [data.shape]
    
    def value_infer(self, input_tensors: List[Tensor]) -> List[np.ndarray]:
        """
        Compute output values for Softmax operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output numpy arrays
        """
        if len(input_tensors) != 1:
            raise ValueError(f"Softmax expects 1 input, got {len(input_tensors)}")
        
        data = input_tensors[0].numpy
        
        if data is None:
            raise ValueError("Softmax input must have a value")
        
        # Get axis from attribute
        axis = self.get_attr("axis", 1)  # Default is 1 for ONNX Softmax
        
        # Convert negative axis to positive
        data_rank = len(data.shape)
        axis = axis if axis >= 0 else axis + data_rank
        
        # Compute softmax along the specified axis
        # Shift data for numerical stability
        shifted_data = data - np.max(data, axis=axis, keepdims=True)
        exp_data = np.exp(shifted_data)
        return [exp_data / np.sum(exp_data, axis=axis, keepdims=True)]

@OPERATOR_REGISTRY.register("Pow")
class PowNode(Node):
    """Implementation of the Pow operator."""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        Infer output shapes for Pow operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        if len(input_tensors) != 2:
            raise ValueError(f"Pow expects 2 inputs, got {len(input_tensors)}")
        
        x = input_tensors[0]
        y = input_tensors[1]
        
        if not x.shape or not y.shape:
            raise ValueError(f"Pow inputs must have shapes: {x.shape}, {y.shape}")
        
        # Implement broadcasting rules
        return [self._broadcast_shapes(x.shape, y.shape)]
    
    def _broadcast_shapes(self, shape_a: List[int], shape_b: List[int]) -> List[int]:
        """Apply numpy broadcasting rules to shapes."""
        # Convert to numpy arrays for easier calculation
        a = np.zeros(shape_a)
        b = np.zeros(shape_b)
        
        # Use numpy broadcasting rules
        try:
            c = np.broadcast_arrays(a, b)[0]
            return list(c.shape)
        except ValueError:
            raise ValueError(f"Cannot broadcast shapes {shape_a} and {shape_b}")

@OPERATOR_REGISTRY.register("ReduceMean")
class ReduceMeanNode(Node):
    """Implementation of the ReduceMean operator."""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        Infer output shapes for ReduceMean operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        if len(input_tensors) != 1:
            raise ValueError(f"ReduceMean expects 1 input, got {len(input_tensors)}")
        
        data = input_tensors[0]
        
        if not data.shape:
            raise ValueError(f"ReduceMean input must have a shape")
        
        # Get axes and keepdims from attributes
        axes = self.get_attr("axes", None)
        keepdims = self.get_attr("keepdims", 1)
        
        if axes is None:
            # Default is to reduce over all axes
            axes = list(range(len(data.shape)))
        
        # Convert negative axes to positive
        data_rank = len(data.shape)
        axes = [axis if axis >= 0 else axis + data_rank for axis in axes]
        
        # Compute output shape
        output_shape = list(data.shape)
        for axis in sorted(axes, reverse=True):
            if keepdims:
                output_shape[axis] = 1
            else:
                output_shape.pop(axis)
        
        return [output_shape]

@OPERATOR_REGISTRY.register("Sqrt")
class SqrtNode(Node):
    """Implementation of the Sqrt operator."""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        Infer output shapes for Sqrt operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        if len(input_tensors) != 1:
            raise ValueError(f"Sqrt expects 1 input, got {len(input_tensors)}")
        
        data = input_tensors[0]
        
        if not data.shape:
            raise ValueError(f"Sqrt input must have a shape")
        
        # Sqrt doesn't change the shape
        return [data.shape]
    
    def value_infer(self, input_tensors: List[Tensor]) -> List[np.ndarray]:
        """
        Compute output values for Sqrt operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output numpy arrays
        """
        if len(input_tensors) != 1:
            raise ValueError(f"Sqrt expects 1 input, got {len(input_tensors)}")
        
        data = input_tensors[0].numpy
        
        if data is None:
            raise ValueError("Sqrt input must have a value")
        
        return [np.sqrt(data)]

@OPERATOR_REGISTRY.register("Div")
class DivNode(Node):
    """Implementation of the Div operator."""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        Infer output shapes for Div operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        if len(input_tensors) != 2:
            raise ValueError(f"Div expects 2 inputs, got {len(input_tensors)}")
        
        a = input_tensors[0]
        b = input_tensors[1]
        
        if not a.shape or not b.shape:
            raise ValueError(f"Div inputs must have shapes: {a.shape}, {b.shape}")
        
        # Implement broadcasting rules
        return [self._broadcast_shapes(a.shape, b.shape)]
    
    def _broadcast_shapes(self, shape_a: List[int], shape_b: List[int]) -> List[int]:
        """Apply numpy broadcasting rules to shapes."""
        # Convert to numpy arrays for easier calculation
        a = np.zeros(shape_a)
        b = np.zeros(shape_b)
        
        # Use numpy broadcasting rules
        try:
            c = np.broadcast_arrays(a, b)[0]
            return list(c.shape)
        except ValueError:
            raise ValueError(f"Cannot broadcast shapes {shape_a} and {shape_b}")
    
    def value_infer(self, input_tensors: List[Tensor]) -> List[np.ndarray]:
        """
        Compute output values for Div operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output numpy arrays
        """
        if len(input_tensors) != 2:
            raise ValueError(f"Div expects 2 inputs, got {len(input_tensors)}")
        
        a = input_tensors[0].numpy
        b = input_tensors[1].numpy
        
        if a is None or b is None:
            raise ValueError("Div inputs must have values")
        
        return [np.divide(a, b)]

@OPERATOR_REGISTRY.register("Sigmoid")
class SigmoidNode(Node):
    """Implementation of the Sigmoid operator."""
    
    def shape_infer(self, input_tensors: List[Tensor]) -> List[List[int]]:
        """
        Infer output shapes for Sigmoid operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        if len(input_tensors) != 1:
            raise ValueError(f"Sigmoid expects 1 input, got {len(input_tensors)}")
        
        data = input_tensors[0]
        
        if not data.shape:
            raise ValueError(f"Sigmoid input must have a shape")
        
        # Sigmoid doesn't change the shape
        return [data.shape]
    
    def value_infer(self, input_tensors: List[Tensor]) -> List[np.ndarray]:
        """
        Compute output values for Sigmoid operator.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output numpy arrays
        """
        if len(input_tensors) != 1:
            raise ValueError(f"Sigmoid expects 1 input, got {len(input_tensors)}")
        
        data = input_tensors[0].numpy
        
        if data is None:
            raise ValueError("Sigmoid input must have a value")
        
        return [1 / (1 + np.exp(-data))]