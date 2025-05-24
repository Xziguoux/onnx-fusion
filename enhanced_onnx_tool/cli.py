"""
Command-line interface for Enhanced ONNX Tool.
"""

import sys
import click
import json
from pathlib import Path

from enhanced_onnx_tool.version import __version__
from enhanced_onnx_tool import infer_shapes, profile_model, optimize_model
from enhanced_onnx_tool.utils.logging import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)

@click.group()
@click.version_option(version=__version__)
def cli():
    """Enhanced ONNX Tool - A robust parser, editor and profiler for ONNX models."""
    pass

@cli.command("infer-shapes")
@click.option("-i", "--input", "input_path", required=True, help="Input ONNX model path")
@click.option("-o", "--output", "output_path", help="Output ONNX model path")
@click.option("--input-shapes", help="Input shapes as JSON dictionary, e.g. '{\"input\": [1, 3, 224, 224]}'")
@click.option("--verbose/--quiet", default=True, help="Verbose output")
def run_infer_shapes(input_path, output_path, input_shapes, verbose):
    """Infer shapes for an ONNX model."""
    try:
        # Parse input shapes if provided
        shapes_dict = None
        if input_shapes:
            shapes_dict = json.loads(input_shapes)
        
        # Set default output path if not provided
        if not output_path:
            input_path_obj = Path(input_path)
            output_path = str(input_path_obj.with_stem(f"{input_path_obj.stem}_shapes"))
        
        # Run shape inference
        logger.info(f"Inferring shapes for {input_path}")
        model = infer_shapes(input_path, shapes_dict, output_path)
        
        # Report success
        if verbose:
            tensor_count = len(model.graph.tensor_map)
            shaped_tensors = sum(1 for t in model.graph.tensor_map.values() if t.shape)
            logger.info(f"Shape inference complete. {shaped_tensors}/{tensor_count} tensors have shapes.")
        
        logger.info(f"Model with inferred shapes saved to {output_path}")
        return 0
    
    except Exception as e:
        logger.error(f"Error during shape inference: {str(e)}")
        if verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1

@cli.command("profile")
@click.option("-i", "--input", "input_path", required=True, help="Input ONNX model path")
@click.option("--input-shapes", help="Input shapes as JSON dictionary, e.g. '{\"input\": [1, 3, 224, 224]}'")
@click.option("-o", "--output", "output_path", help="Output profile report path (JSON)")
@click.option("--per-node/--summary", default=False, help="Include per-node statistics")
@click.option("--verbose/--quiet", default=True, help="Verbose output")
def run_profile(input_path, input_shapes, output_path, per_node, verbose):
    """Profile an ONNX model for MACs, parameters, etc."""
    try:
        # Parse input shapes if provided
        shapes_dict = None
        if input_shapes:
            shapes_dict = json.loads(input_shapes)
        
        # Run profiling
        logger.info(f"Profiling model {input_path}")
        profile_results = profile_model(input_path, shapes_dict)
        
        # Print summary
        if verbose:
            logger.info(f"Total MACs: {profile_results['total_macs']:,}")
            logger.info(f"Total Parameters: {profile_results['total_params']:,}")
            
            if per_node and 'per_node' in profile_results:
                logger.info("\nTop 10 nodes by MACs:")
                # Sort nodes by MACs and print top 10
                sorted_nodes = sorted(
                    profile_results['per_node'].items(), 
                    key=lambda x: x[1]['macs'], 
                    reverse=True
                )
                for i, (name, stats) in enumerate(sorted_nodes[:10]):
                    logger.info(f"{i+1}. {name} ({stats['op_type']}): {stats['macs']:,} MACs")
        
        # Save report if requested
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(profile_results, f, indent=2)
            logger.info(f"Profile report saved to {output_path}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during profiling: {str(e)}")
        if verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1

@cli.command("optimize")
@click.option("-i", "--input", "input_path", required=True, help="Input ONNX model path")
@click.option("-o", "--output", "output_path", required=True, help="Output ONNX model path")
@click.option("--level", type=int, default=1, help="Optimization level (1-3)")
@click.option("--constant-folding/--no-constant-folding", default=True, help="Enable constant folding")
@click.option("--verbose/--quiet", default=True, help="Verbose output")
def run_optimize(input_path, output_path, level, constant_folding, verbose):
    """Optimize an ONNX model."""
    try:
        # Set optimization options
        options = {
            "level": level,
            "constant_folding": constant_folding,
        }
        
        # Run optimization
        logger.info(f"Optimizing model {input_path} (level {level})")
        model = optimize_model(input_path, output_path, **options)
        
        # Report success
        if verbose:
            node_count = len(model.graph.nodes)
            logger.info(f"Optimization complete. Model has {node_count} nodes.")
        
        logger.info(f"Optimized model saved to {output_path}")
        return 0
    
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        if verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1

def main():
    """Entry point for the CLI."""
    return cli(prog_name="enhanced-onnx-tool")

if __name__ == "__main__":
    sys.exit(main())
