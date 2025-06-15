#!/usr/bin/env python3
"""
Run GradCAM Analysis for Duck Classification Model
Analyzes the trained model using Grad-CAM to understand which features the model focuses on

Usage:
    python run_gradcam_analysis.py [--data-path PATH] [--output-dir PATH] [--model-path PATH] [--params-file PATH]

Example:
    python run_gradcam_analysis.py --data-path images/Phone --output-dir output --model-path output/best_model_*/model/best_duck_classifier_f1_optimized.h5
"""

import sys
import os
import argparse
import glob
from pathlib import Path

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run GradCAM Analysis for Duck Classification Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="images/Phone",
        help="Path to the directory containing duck images organized by species"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output",
        help="Directory where all analysis outputs will be saved"
    )
    
    parser.add_argument(
        "--model-path", 
        type=str, 
        default=None,
        help="Path to the trained model file (.h5). If not specified, will auto-detect from output directory"
    )
    
    parser.add_argument(
        "--params-file", 
        type=str, 
        default="src/best_model.json",
        help="JSON file containing the best hyperparameters"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = {
        'tensorflow': 'tensorflow',
        'numpy': 'numpy', 
        'matplotlib.pyplot': 'matplotlib',
        'PIL': 'pillow',
        'tf_keras_vis.gradcam_plus_plus': 'tf-keras-vis'
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing dependencies:")
        for package in missing:
            print(f"   • {package}")
        print("\n📥 Install missing packages with:")
        print(f"   pip install {' '.join(missing)}")
        return False
    else:
        print("✅ All dependencies found")
        return True

def find_model_file(output_dir):
    """Auto-detect the most recent model file in the output directory"""
    patterns = [
        os.path.join(output_dir, "**/model/best_duck_classifier_f1_optimized.h5"),
        os.path.join(output_dir, "**/model/*.h5"),
        os.path.join(output_dir, "**/*.h5")
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            # Return the most recently modified file
            return max(matches, key=os.path.getmtime)
    
    return None

def validate_paths(args):
    """Validate that required paths exist and are accessible"""
    errors = []
    
    # Check data path
    if not os.path.exists(args.data_path):
        errors.append(f"Data path '{args.data_path}' does not exist!")
    elif not os.path.isdir(args.data_path):
        errors.append(f"Data path '{args.data_path}' is not a directory!")
    
    # Auto-detect model path if not provided
    if args.model_path is None:
        args.model_path = find_model_file(args.output_dir)
        if args.model_path is None:
            errors.append(f"No model file found in '{args.output_dir}'. Please specify --model-path")
    
    # Check model path
    if args.model_path and not os.path.exists(args.model_path):
        errors.append(f"Model file '{args.model_path}' does not exist!")
    
    # Check params file
    if not os.path.exists(args.params_file):
        print(f"Warning: Parameters file '{args.params_file}' not found. Default parameters will be used.")
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create output directory '{args.output_dir}': {e}")
    
    return errors

def main():
    """Main function to run GradCAM analysis"""
    
    # Parse arguments
    args = parse_arguments()
    
    print("="*70)
    print("🔍 GRADCAM ANALYSIS FOR DUCK CLASSIFICATION MODEL")
    print("="*70)
    
    if args.verbose:
        print(f"Data path: {args.data_path}")
        print(f"Output directory: {args.output_dir}")
        print(f"Model path: {args.model_path}")
        print(f"Parameters file: {args.params_file}")
        print("-" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Validate paths
    validation_errors = validate_paths(args)
    if validation_errors:
        print("❌ VALIDATION ERRORS:")
        for error in validation_errors:
            print(f"   • {error}")
        print("\n💡 Please fix the above issues and try again.")
        return 1
    
    # Display configuration
    print("✅ Configuration:")
    print(f"   📁 Data path: {args.data_path}")
    print(f"   📂 Output directory: {args.output_dir}")
    print(f"   🤖 Model file: {args.model_path}")
    print(f"   ⚙️  Parameters file: {args.params_file}")
    
    # Import and run analysis
    try:
        from gradcam_best_model_analysis import GradCAMBestModelAnalyzer
        
        print("\n🎯 Initializing GradCAM analyzer...")
        analyzer = GradCAMBestModelAnalyzer(
            best_model_path=args.model_path,
            best_params_file=args.params_file,
            data_path=args.data_path,
            output_base_dir=args.output_dir
        )
        
        print("🔄 Running complete GradCAM analysis...")
        print("   This may take several minutes depending on your dataset size...")
        analyzer.run_complete_analysis()
        
        print("\n" + "="*70)
        print("🎉 GRADCAM ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📁 Results saved in: {analyzer.output_dir}")
        print("📊 Generated visualizations:")
        print("   • Most confident correct predictions per class")
        print("   • Most confident incorrect predictions per class")
        print("   • Class-specific GradCAM heatmaps")
        
        print("\n💡 Next steps:")
        print("   • Review the GradCAM visualizations to understand model focus")
        print("   • Analyze incorrect predictions to identify model biases")
        print("   • Use insights to improve data collection or model architecture")
        
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure the gradcam_best_model_analysis.py file is in the same directory")
        return 1
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        if args.verbose:
            import traceback
            print("\n🔧 Full error traceback:")
            traceback.print_exc()
        else:
            print("💡 Use --verbose flag for detailed error information")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 