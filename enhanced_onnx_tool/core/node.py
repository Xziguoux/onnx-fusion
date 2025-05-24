"""
Node class for Enhanced ONNX Tool.
"""

import onnx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Type
from collections import OrderedDict

from enhanced_onnx_tool.utils.logging import get_logger

logger = get_logger(__name__)

class Node:
    """Represents an operator node in the computation graph."""
    
    def __init__(self, name: str = "", op_type: str = ""):
        """
        Initialize a node.
        
        Args:
            name: Node name
            op_type: Operator type
        """
        self.name = name
        self.op_type = op_type
        self.inputs: List[str] = []  # Input tensor names
        self.outputs: List[str] = []  # Output tensor names
        self.attributes: Dict[str, Any] = OrderedDict()  # Operator attributes
        self.domain = ""  # Operator domain (default is empty for ai.onnx)
    
    @classmethod
    def from_onnx(cls, node_proto: onnx.NodeProto) -> 'Node':
        """
        Create a Node from an ONNX NodeProto.
        
        Args:
            node_proto: ONNX NodeProto object
            
        Returns:
            New Node instance
        """
        # Import here to avoid circular imports
        from enhanced_onnx_tool.operators.registry import OPERATOR_REGISTRY
        
        # Get specialized node class if available
        node_cls = OPERATOR_REGISTRY.get_node_class(node_proto.op_type, node_proto.domain)
        
        # Create node
        node = node_cls(node_proto.name, node_proto.op_type)
        node.domain = node_proto.domain
        
        # Set inputs and outputs
        node.inputs = list(node_proto.input)
        node.outputs = list(node_proto.output)
        
        # Parse attributes
        for attr in node_proto.attribute:
            node.attributes[attr.name] = Node._parse_attribute(attr)
        
        return node
    
    def to_onnx(self) -> onnx.NodeProto:
        """
        Convert the Node to an ONNX NodeProto.
        
        Returns:
            ONNX NodeProto
        """
        node_proto = onnx.NodeProto()
        node_proto.name = self.name
        node_proto.op_type = self.op_type
        node_proto.domain = self.domain
        
        # Set inputs and outputs
        node_proto.input.extend(self.inputs)
        node_proto.output.extend(self.outputs)
        
        # Set attributes
        for name, value in self.attributes.items():
            attr = Node._make_attribute(name, value)
            node_proto.attribute.append(attr)
        
        return node_proto
    
    def shape_infer(self, input_tensors: List['Tensor']) -> List[List[int]]:
        """
        Infer output shapes based on input shapes.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output shapes
        """
        # Try to find a specialized implementation in the registry
        from enhanced_onnx_tool.operators.registry import OPERATOR_REGISTRY
        
        specialized_cls = OPERATOR_REGISTRY.get_node_class(self.op_type, self.domain)
        
        # If the specialized class is not this base class, delegate to it
        if specialized_cls != Node and specialized_cls != self.__class__:
            specialized_node = specialized_cls(self.name, self.op_type)
            specialized_node.inputs = self.inputs
            specialized_node.outputs = self.outputs
            specialized_node.attributes = self.attributes
            specialized_node.domain = self.domain
            
            return specialized_node.shape_infer(input_tensors)
        
        # Default implementation if no specialized implementation found
        logger.warning(f"No shape inference implemented for {self.op_type}, using passthrough")
        
        # Passthrough shapes for identity-like operators
        if len(input_tensors) == 1 and self.is_identity_like():
            return [input_tensors[0].shape]
        
        # Return None shapes for outputs if we can't infer
        return [None] * len(self.outputs)
    
    def value_infer(self, input_tensors: List['Tensor']) -> List[np.ndarray]:
        """
        Compute actual output values based on input values.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            List of output numpy arrays
        """
        # Try to find a specialized implementation in the registry
        from enhanced_onnx_tool.operators.registry import OPERATOR_REGISTRY
        
        specialized_cls = OPERATOR_REGISTRY.get_node_class(self.op_type, self.domain)
        
        # If the specialized class is not this base class, delegate to it
        if specialized_cls != Node and specialized_cls != self.__class__:
            specialized_node = specialized_cls(self.name, self.op_type)
            specialized_node.inputs = self.inputs
            specialized_node.outputs = self.outputs
            specialized_node.attributes = self.attributes
            specialized_node.domain = self.domain
            
            return specialized_node.value_infer(input_tensors)
        
        # Default implementation if no specialized implementation found
        logger.warning(f"No value inference implemented for {self.op_type}")
        
        # Passthrough values for identity-like operators
        if len(input_tensors) == 1 and self.is_identity_like() and input_tensors[0].numpy is not None:
            return [input_tensors[0].numpy]
        
        # Return None values for outputs if we can't infer
        return [None] * len(self.outputs)
    
    def profile(self, input_tensors: List['Tensor'], output_tensors: List['Tensor']) -> Tuple[int, int]:
        """
        Profile the computational complexity of this node.
        
        Args:
            input_tensors: List of input tensors
            output_tensors: List of output tensors
            
        Returns:
            Tuple of (MACs, parameters)
        """
        # Try to find a specialized implementation in the registry
        from enhanced_onnx_tool.operators.registry import OPERATOR_REGISTRY
        
        specialized_cls = OPERATOR_REGISTRY.get_node_class(self.op_type, self.domain)
        
        # If the specialized class is not this base class, delegate to it
        if specialized_cls != Node and specialized_cls != self.__class__:
            specialized_node = specialized_cls(self.name, self.op_type)
            specialized_node.inputs = self.inputs
            specialized_node.outputs = self.outputs
            specialized_node.attributes = self.attributes
            specialized_node.domain = self.domain
            
            return specialized_node.profile(input_tensors, output_tensors)
        
        # Default implementation if no specialized implementation found
        logger.debug(f"No profiling implemented for {self.op_type}, assuming zero MACs")
        
        # Count parameters in weights
        params = 0
        for tensor in input_tensors:
            if tensor.is_constant:
                params += np.prod(tensor.shape) if tensor.shape else 0
        
        # Return zeroes by default
        return 0, params
    
    def is_identity_like(self) -> bool:
        """
        Check if this node behaves like an identity operation.
        
        Returns:
            True if node is identity-like
        """
        # List of known identity-like operators
        identity_ops = {'Identity', 'Reshape', 'Flatten', 'Squeeze', 'Unsqueeze', 'Transpose', 'Cast'}
        return self.op_type in identity_ops
    
    def get_attr(self, name: str, default: Any = None) -> Any:
        """
        Get an attribute value.
        
        Args:
            name: Attribute name
            default: Default value if attribute doesn't exist
            
        Returns:
            Attribute value or default
        """
        return self.attributes.get(name, default)
    
    def set_attr(self, name: str, value: Any) -> None:
        """
        Set an attribute value.
        
        Args:
            name: Attribute name
            value: Attribute value
        """
        self.attributes[name] = value
    
    @staticmethod
    def _parse_attribute(attr: onnx.AttributeProto) -> Any:
        """
        Parse an ONNX attribute.
        
        Args:
            attr: ONNX AttributeProto
            
        Returns:
            Python value for the attribute
        """
        if attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode('utf-8')
        elif attr.type == onnx.AttributeProto.TENSOR:
            from enhanced_onnx_tool.core.tensor import Tensor
            return Tensor.from_onnx_tensor(attr.t).numpy
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == onnx.AttributeProto.STRINGS:
            return [s.decode('utf-8') for s in attr.strings]
        elif attr.type == onnx.AttributeProto.TENSORS:
            from enhanced_onnx_tool.core.tensor import Tensor
            return [Tensor.from_onnx_tensor(t).numpy for t in attr.tensors]
        elif attr.type == onnx.AttributeProto.GRAPH:
            # We don't parse graphs yet
            logger.warning(f"Graph attribute '{attr.name}' not fully parsed")
            return None
        else:
            logger.warning(f"Unknown attribute type {attr.type} for attribute '{attr.name}'")
            return None
    
    @staticmethod
    def _make_attribute(name: str, value: Any) -> onnx.AttributeProto:
        """
        Create an ONNX attribute.
        
        Args:
            name: Attribute name
            value: Attribute value
            
        Returns:
            ONNX AttributeProto
        """
        attr = onnx.AttributeProto()
        attr.name = name
        
        if isinstance(value, float):
            attr.type = onnx.AttributeProto.FLOAT
            attr.f = value
        elif isinstance(value, int):
            attr.type = onnx.AttributeProto.INT
            attr.i = value
        elif isinstance(value, str):
            attr.type = onnx.AttributeProto.STRING
            attr.s = value.encode('utf-8')
        elif isinstance(value, np.ndarray):
            attr.type = onnx.AttributeProto.TENSOR
            from enhanced_onnx_tool.core.tensor import Tensor
            tensor = Tensor("", None, list(value.shape), value)
            attr.t.CopyFrom(tensor.to_onnx_tensor())
        elif isinstance(value, list):
            if value and isinstance(value[0], float):
                attr.type = onnx.AttributeProto.FLOATS
                attr.floats.extend(value)
            elif value and isinstance(value[0], int):
                attr.type = onnx.AttributeProto.INTS
                attr.ints.extend(value)
            elif value and isinstance(value[0], str):
                attr.type = onnx.AttributeProto.STRINGS
                attr.strings.extend([s.encode('utf-8') for s in value])
            elif value and isinstance(value[0], np.ndarray):
                attr.type = onnx.AttributeProto.TENSORS
                from enhanced_onnx_tool.core.tensor import Tensor
                for v in value:
                    tensor = Tensor("", None, list(v.shape), v)
                    attr.tensors.append(tensor.to_onnx_tensor())
            else:
                raise ValueError(f"Unsupported list type for attribute '{name}'")
        else:
            raise ValueError(f"Unsupported value type {type(value)} for attribute '{name}'")
        
        return attr
    
    def __repr__(self) -> str:
        """String representation of the node."""
        return f"Node(name='{self.name}', op_type='{self.op_type}', inputs={len(self.inputs)}, outputs={len(self.outputs)})"
