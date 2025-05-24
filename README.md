# onnx-fusion Enhanced ONNX Tool

An enhanced parser, editor, and profiler tool for ONNX models with robust shape inference capabilities.

## Features

- **Robust Shape Inference**: Reliable shape inference that works where standard ONNX tools fail
- **Shape Engine & Compute Graph Separation**: Clear separation between shape calculations and compute operations
- **Model Profiling**: Analyze model complexity, MACs, and memory usage
- **Model Optimization**: Constant folding, memory optimization, and quantization support
- **Extensible Architecture**: Easily add custom operators and analysis tools

## Installation

```bash
# Install from PyPI
pip install enhanced-onnx-tool

# Install from source
git clone https://github.com/yourusername/enhanced-onnx-tool.git
cd enhanced-onnx-tool
pip install -e .
```

## Usage

### Command Line

```bash
# Basic shape inference
enhanced-onnx-tool infer-shapes -i model.onnx -o model_with_shapes.onnx

# Profile model
enhanced-onnx-tool profile -i model.onnx --input-shapes "input:1,3,224,224"

# Optimize model
enhanced-onnx-tool optimize -i model.onnx -o optimized_model.onnx --level 2
```

### Python API

```python
import enhanced_onnx_tool as eot

# Load a model
model = eot.Model("model.onnx")

# Infer shapes
model.infer_shapes({"input": [1, 3, 224, 224]})

# Profile model
profile_result = model.profile()
print(f"Total MACs: {profile_result['total_macs']}")

# Optimize and save
model.optimize(level=2)
model.save("optimized_model.onnx")
```

## Examples

See the `examples/` directory for more detailed usage examples.

## Documentation

Detailed documentation is available at [https://enhanced-onnx-tool.readthedocs.io](https://enhanced-onnx-tool.readthedocs.io).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
