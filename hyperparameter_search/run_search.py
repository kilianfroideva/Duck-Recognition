#!/usr/bin/env python3
"""
Simple runner script for hyperparameter search
Usage: python run_search.py [strategy]

Available strategies:
- quick_search: Fast search focusing on most important parameters
- thorough_search: Comprehensive search of all parameters  
- architecture_focus: Focus on model architecture parameters
- regularization_focus: Focus on regularization parameters
- default: Standard search (30 trials)
"""

import sys
import argparse
from hyperparameter_optimizer import HyperparameterOptimizer
from config import DATA_PATH, SEARCH_STRATEGIES, SEARCH_CONFIG

def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter search for duck classification')
    parser.add_argument('--strategy', type=str, default='default',
                       choices=['quick_search', 'thorough_search', 'architecture_focus', 
                               'regularization_focus', 'default'],
                       help='Search strategy to use')
    parser.add_argument('--n_calls', type=int, help='Override number of trials')
    parser.add_argument('--max_epochs', type=int, help='Override max epochs per trial')
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Path to image data')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting hyperparameter search with '{args.strategy}' strategy")
    
    # Create optimizer with strategy-specific settings
    optimizer = HyperparameterOptimizer(data_path=args.data_path)
    
    # Apply strategy settings
    if args.strategy != 'default' and args.strategy in SEARCH_STRATEGIES:
        strategy_config = SEARCH_STRATEGIES[args.strategy]
        optimizer.n_calls = strategy_config['n_calls']
        optimizer.max_epochs = strategy_config['max_epochs']
        
        # If strategy focuses on specific parameters, you could modify search space here
        print(f"📋 Strategy settings:")
        print(f"   - Trials: {optimizer.n_calls}")
        print(f"   - Max epochs: {optimizer.max_epochs}")
        if 'focus_on' in strategy_config:
            print(f"   - Focus on: {strategy_config['focus_on']}")
    
    # Apply command line overrides
    if args.n_calls:
        optimizer.n_calls = args.n_calls
        print(f"🔧 Override: Using {args.n_calls} trials")
    
    if args.max_epochs:
        optimizer.max_epochs = args.max_epochs
        print(f"🔧 Override: Using {args.max_epochs} max epochs")
    
    # Run the search
    try:
        optimizer.run_search()
        print("\n🎉 Search completed successfully!")
        print(f"📊 Results saved to: {optimizer.run_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Search interrupted by user")
        print("💾 Progress has been saved")
    except Exception as e:
        print(f"\n❌ Error during search: {e}")
        print("💾 Partial results may have been saved")


if __name__ == "__main__":
    main() 