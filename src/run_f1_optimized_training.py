#!/usr/bin/env python3
"""
Run F1-Optimized Duck Classification Training
Uses the best hyperparameters from hyperparameter search to train a model optimized for F1-macro score

Usage:
    python run_f1_optimized_training.py [--data-path PATH] [--output-dir PATH] [--params-file PATH]

Example:
    python run_f1_optimized_training.py --data-path images/Phone --output-dir output --params-file src/best_model.json
"""

import os
import sys
import argparse
from pathlib import Path
from duck_classification_complete import DuckClassifierOptimized

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run F1-Optimized Duck Classification Training",
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
        help="Directory where all training outputs will be saved"
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

def validate_paths(args):
    """Validate that required paths exist and are accessible"""
    errors = []
    
    # Check data path
    if not os.path.exists(args.data_path):
        errors.append(f"Data path '{args.data_path}' does not exist!")
    elif not os.path.isdir(args.data_path):
        errors.append(f"Data path '{args.data_path}' is not a directory!")
    
    # Check if data path has subdirectories (species folders)
    if os.path.exists(args.data_path):
        subdirs = [d for d in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, d))]
        if not subdirs:
            errors.append(f"Data path '{args.data_path}' contains no subdirectories! Images should be organized by species.")
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create output directory '{args.output_dir}': {e}")
    
    # Check params file
    if not os.path.exists(args.params_file):
        print(f"Warning: Parameters file '{args.params_file}' not found. Default parameters will be used.")
    
    return errors

def main():
    """Main function to run the optimized duck classification pipeline"""
    
    # Parse arguments
    args = parse_arguments()
    
    print("="*70)
    print("F1-MACRO OPTIMIZED DUCK SPECIES CLASSIFICATION")
    print("="*70)
    
    if args.verbose:
        print(f"Data path: {args.data_path}")
        print(f"Output directory: {args.output_dir}")
        print(f"Parameters file: {args.params_file}")
        print("-" * 50)
    
    # Validate paths
    validation_errors = validate_paths(args)
    if validation_errors:
        print("❌ VALIDATION ERRORS:")
        for error in validation_errors:
            print(f"   • {error}")
        print("\n💡 Please fix the above issues and try again.")
        return 1
    
    try:
        # Initialize classifier
        print("🚀 Initializing classifier...")
        classifier = DuckClassifierOptimized(
            data_path=args.data_path, 
            output_dir=args.output_dir,
            best_params_file=args.params_file
        )
        
        # Display configuration
        print(f"✅ Configuration:")
        print(f"   📁 Data path: {args.data_path}")
        print(f"   📂 Output directory: {args.output_dir}")
        print(f"   ⚙️  Parameters file: {args.params_file}")
        
        # Run complete pipeline
        print("\n🔄 Starting complete training pipeline...")
        print("   This may take several minutes depending on your hardware...")
        
        model, history, predictions, true_labels, cv_stats = classifier.run_complete_pipeline()
        
        print("\n" + "="*70)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📁 Results saved in: {classifier.run_dir}")
        print(f"🎯 Cross-validation F1-Macro: {cv_stats['mean_f1_macro']:.4f} ± {cv_stats['std_f1_macro']:.4f}")
        print(f"📈 Model saved as: {classifier.run_dir}/model/")
        print(f"📊 Plots available in: {classifier.run_dir}/plots/")
        
        print("\n💡 Next steps:")
        print("   • Review the training plots for model performance")
        print("   • Run GradCAM analysis to understand model decisions")
        print("   • Test the model on new images")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        if args.verbose:
            import traceback
            print("\n🔧 Full error traceback:")
            traceback.print_exc()
        else:
            print("💡 Use --verbose flag for detailed error information")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 