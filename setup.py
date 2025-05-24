from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="enhanced-onnx-tool",
    version="0.1.0",
    author="ONNX Tool Team",
    author_email="example@example.com",
    description="Enhanced ONNX parser, editor and profiler tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/enhanced-onnx-tool",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.16.0",
        "onnx>=1.8.0",
        "protobuf>=3.12.0",
        "click>=7.0",
        "tqdm>=4.45.0",
        "colorama>=0.4.3",
        "networkx>=2.5",
    ],
    entry_points={
        "console_scripts": [
            "enhanced-onnx-tool=enhanced_onnx_tool.cli:main",
        ],
    },
)
