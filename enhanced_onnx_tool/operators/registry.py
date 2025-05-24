"""
Operator registry for Enhanced ONNX Tool.
"""

from typing import Dict, Type, Callable, Any

class OperatorRegistry:
    """
    Registry for ONNX operators.
    """
    
    def __init__(self):
        """Initialize an empty registry."""
        self._operators = {}
    
    def register(self, op_type: str) -> Callable:
        """
        Decorator to register an operator implementation.
        
        Args:
            op_type: ONNX operator type name
            
        Returns:
            Decorator function
        """
        def decorator(cls):
            self._operators[op_type] = cls
            return cls
        return decorator
    
    def get(self, op_type: str) -> Type:
        """
        Get operator implementation by type.
        
        Args:
            op_type: ONNX operator type name
            
        Returns:
            Operator class
            
        Raises:
            KeyError: If operator is not registered
        """
        if op_type not in self._operators:
            raise KeyError(f"Operator '{op_type}' not registered")
        return self._operators[op_type]
    
    def get_node_class(self, op_type: str, domain: str = "") -> Type:
        """
        Get operator implementation by type and domain.
        
        Args:
            op_type: ONNX operator type name
            domain: Operator domain (default: "")
            
        Returns:
            Operator class
        """
        try:
            # 尝试获取注册的算子
            return self.get(op_type)
        except KeyError:
            # 如果算子未注册，使用GenericNode作为后备
            from enhanced_onnx_tool.operators.base import GenericNode
            import logging
            logging.getLogger(__name__).warning(f"使用通用节点处理未注册的算子: {op_type} (domain: {domain})")
            return GenericNode
    
    def contains(self, op_type: str) -> bool:
        """
        Check if operator is registered.
        
        Args:
            op_type: ONNX operator type name
            
        Returns:
            True if operator is registered, False otherwise
        """
        return op_type in self._operators
    
    def list_operators(self) -> Dict[str, Type]:
        """
        Get a dictionary of all registered operators.
        
        Returns:
            Dictionary mapping operator names to their implementations
        """
        return self._operators.copy()
    
    def __len__(self) -> int:
        """Get number of registered operators."""
        return len(self._operators)

# Global operator registry
OPERATOR_REGISTRY = OperatorRegistry()
