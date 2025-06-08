#!/usr/bin/env python3
"""
Run F1-Optimized Duck Classification Training
Uses the best hyperparameters from hyperparameter search to train a model optimized for F1-macro score
"""

import os
import sys
from duck_classification_complete import DuckClassifierOptimized

def main():
    """Main function to run the optimized duck classification pipeline"""
    DATA_PATH = "images/Phone"
    OUTPUT_DIR = "output"
    BEST_PARAMS_FILE = "best_model.json"
    
    print("="*60)
    print("F1-MACRO OPTIMIZED DUCK SPECIES CLASSIFICATION")
    print("="*60)
    
    # Check if data path exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data path '{DATA_PATH}' does not exist!")
        print("Please ensure your duck images are in the 'images/Phone' directory.")
        return
    
    # Check if best parameters file exists
    if not os.path.exists(BEST_PARAMS_FILE):
        print(f"Warning: {BEST_PARAMS_FILE} not found. Using default parameters.")
    
    try:
        # Initialize classifier
        print(f"Initializing classifier...")
        classifier = DuckClassifierOptimized(
            data_path=DATA_PATH, 
            output_dir=OUTPUT_DIR,
            best_params_file=BEST_PARAMS_FILE
        )
        
        print(f"Data path: {DATA_PATH}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Best parameters file: {BEST_PARAMS_FILE}")
        
        # Run complete pipeline
        print("\nStarting complete pipeline...")
        model, history, predictions, true_labels, cv_stats = classifier.run_complete_pipeline()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved in: {classifier.run_dir}")
        print(f"Cross-validation F1-Macro: {cv_stats['mean_f1_macro']:.4f} ± {cv_stats['std_f1_macro']:.4f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 