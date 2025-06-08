"""
Configuration file for hyperparameter search
Modify these settings to customize your search
"""

# Data paths
DATA_PATH = "images/Phone"
OUTPUT_DIR = "hyperparameter_search"

# Search configuration
SEARCH_CONFIG = {
    'n_calls': 30,           # Number of hyperparameter combinations to try
    'max_epochs': 15,        # Max epochs per trial (reduced for speed)
    'final_epochs': 30,      # Epochs for final best model training
    'patience': 4,           # Early stopping patience during search
    'final_patience': 7,     # Early stopping patience for final model
    'n_initial_points': 5,   # Random exploration trials before Bayesian optimization
    'acq_func': 'EI',       # Acquisition function: 'EI', 'PI', 'LCB'
    'random_state': 42,
}

# Hyperparameter search spaces (prioritized by expected impact)
SEARCH_SPACES = {
    # 1. HIGHEST IMPACT PARAMETERS (explore first)
    'learning_rate': {
        'type': 'Real',
        'low': 1e-5,
        'high': 1e-1,
        'prior': 'log-uniform',
        'priority': 1
    },
    
    'fine_tuning': {
        'type': 'Categorical',
        'categories': ['frozen', 'partial', 'full'],
        'priority': 1
    },
    
    # 2. HIGH IMPACT PARAMETERS
    'dropout1': {
        'type': 'Real',
        'low': 0.1,
        'high': 0.6,
        'priority': 2
    },
    
    'dropout2': {
        'type': 'Real',
        'low': 0.1,
        'high': 0.5,
        'priority': 2
    },
    
    'dense_units': {
        'type': 'Integer',
        'low': 64,
        'high': 512,
        'priority': 2
    },
    
    # 3. MEDIUM IMPACT PARAMETERS
    'batch_size': {
        'type': 'Categorical',
        'categories': [16, 32, 64],
        'priority': 3
    },
    
    'optimizer': {
        'type': 'Categorical',
        'categories': ['adam', 'rmsprop', 'sgd'],
        'priority': 3
    },
    
    'use_batch_norm': {
        'type': 'Categorical',
        'categories': [False, True],
        'priority': 3
    },
    
    # 4. LOWER IMPACT PARAMETERS
    'l2_reg': {
        'type': 'Real',
        'low': 1e-6,
        'high': 1e-2,
        'prior': 'log-uniform',
        'priority': 4
    },
}

# Advanced search strategies
SEARCH_STRATEGIES = {
    'quick_search': {
        'n_calls': 15,
        'max_epochs': 10,
        'focus_on': ['learning_rate', 'fine_tuning', 'dropout1']
    },
    
    'thorough_search': {
        'n_calls': 50,
        'max_epochs': 20,
        'focus_on': 'all'
    },
    
    'architecture_focus': {
        'n_calls': 25,
        'max_epochs': 15,
        'focus_on': ['dense_units', 'fine_tuning', 'use_batch_norm', 'dropout1', 'dropout2']
    },
    
    'regularization_focus': {
        'n_calls': 20,
        'max_epochs': 15,
        'focus_on': ['dropout1', 'dropout2', 'l2_reg', 'batch_size']
    }
}

# Model evaluation settings
EVALUATION_CONFIG = {
    'validation_subset_size': 200,  # Size of validation subset for quick F1 evaluation
    'use_f1_for_optimization': True,  # Use F1 score instead of accuracy for optimization
    'save_frequency': 1,  # Save results after every N trials
}

# Visualization settings
PLOT_CONFIG = {
    'save_plots': True,
    'plot_dpi': 300,
    'figure_size': (15, 10),
    'style': 'seaborn',
}

# Error handling settings
ERROR_CONFIG = {
    'max_retries': 2,  # Number of retries on model training failure
    'fallback_score': 0.0,  # Score to assign on complete failure
    'continue_on_error': True,  # Continue search even if some trials fail
} 