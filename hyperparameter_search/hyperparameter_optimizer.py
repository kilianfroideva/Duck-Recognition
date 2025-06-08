#!/usr/bin/env python3
"""
Intelligent Hyperparameter Search for Duck Classification
Features:
- Smart parameter search prioritization
- Real-time progress saving
- Error handling and recovery
- Time estimation with progress bar
- Bayesian optimization for efficiency
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import the main classifier
sys.path.append(str(Path(__file__).parent.parent))

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import base classifier
from duck_classification_complete import DuckClassifier

class HyperparameterOptimizer:
    def __init__(self, data_path="images/Phone", base_output_dir="hyperparameter_search"):
        self.data_path = data_path
        self.base_output_dir = Path(base_output_dir)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_output_dir / f"hp_search_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.run_dir / "models").mkdir(exist_ok=True)
        (self.run_dir / "best_model").mkdir(exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        
        # Initialize base classifier for data loading
        self.base_classifier = DuckClassifier(data_path=data_path)
        self.label_names = self.base_classifier.label_names
        
        # Load and split data once
        print("Loading dataset...")
        self.train_df, self.val_df, self.test_df = self.base_classifier.load_and_split_data()
        print(f"Dataset loaded: {len(self.train_df)} train, {len(self.val_df)} val, {len(self.test_df)} test")
        
        # Search configuration
        self.n_calls = 30  # Number of hyperparameter combinations to try
        self.max_epochs = 15  # Reduced for faster search
        self.patience = 4
        
        # Results tracking
        self.search_results = []
        self.best_score = 0.0
        self.best_params = None
        self.search_start_time = None
        self.completed_trials = 0
        
        # Initialize search space
        self.search_space = self.define_search_space()
        
    def define_search_space(self):
        """Define the hyperparameter search space, prioritized by impact"""
        return [
            # 1. Learning Rate (HIGHEST IMPACT)
            Real(1e-5, 1e-1, prior='log-uniform', name='learning_rate'),
            
            # 2. Dropout rates (HIGH IMPACT)
            Real(0.1, 0.6, name='dropout1'),
            Real(0.1, 0.5, name='dropout2'),
            
            # 3. Dense layer architecture (HIGH IMPACT)
            Integer(64, 512, name='dense_units'),
            Categorical([False, True], name='use_batch_norm'),
            
            # 4. Batch size (MEDIUM IMPACT)
            Categorical([16, 32, 64], name='batch_size'),
            
            # 5. Optimizer choice (MEDIUM IMPACT)
            Categorical(['adam', 'rmsprop', 'sgd'], name='optimizer'),
            
            # 6. Fine-tuning strategy (HIGH IMPACT)
            Categorical(['frozen', 'partial', 'full'], name='fine_tuning'),
            
            # 7. L2 regularization
            Real(1e-6, 1e-2, prior='log-uniform', name='l2_reg'),
        ]
    
    def create_model_from_params(self, params: Dict[str, Any]):
        """Create model from hyperparameters"""
        # Use MobileNetV2 as specified in requirements
        base_model = MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
        
        # Set trainability based on fine-tuning strategy
        if params['fine_tuning'] == 'frozen':
            base_model.trainable = False
        elif params['fine_tuning'] == 'partial':
            base_model.trainable = True
            # Freeze first 80% of layers
            freeze_until = int(len(base_model.layers) * 0.8)
            for layer in base_model.layers[:freeze_until]:
                layer.trainable = False
        else:  # full fine-tuning
            base_model.trainable = True
        
        # Build model
        layers = [base_model, GlobalAveragePooling2D()]
        
        # Add dropout
        layers.append(Dropout(params['dropout1']))
        
        # Add dense layer with optional batch normalization
        layers.append(Dense(params['dense_units'], activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg'])))
        
        if params['use_batch_norm']:
            layers.append(BatchNormalization())
        
        layers.append(Dropout(params['dropout2']))
        layers.append(Dense(len(self.label_names), activation='softmax'))
        
        model = Sequential(layers)
        
        # Get optimizer
        if params['optimizer'] == 'adam':
            optimizer = Adam(learning_rate=params['learning_rate'])
        elif params['optimizer'] == 'rmsprop':
            optimizer = RMSprop(learning_rate=params['learning_rate'])
        else:  # sgd
            optimizer = SGD(learning_rate=params['learning_rate'], momentum=0.9)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_and_evaluate(self, params: Dict[str, Any]) -> float:
        """Train model with given parameters and return validation F1 score"""
        try:
            # Create model
            model = self.create_model_from_params(params)
            
            # Prepare data generators
            batch_size = params['batch_size']
            train_steps = len(self.train_df) // batch_size
            val_steps = len(self.val_df) // batch_size
            
            train_gen = self.base_classifier.data_generator(self.train_df, augment=True, batch_size=batch_size)
            val_gen = self.base_classifier.data_generator(self.val_df, augment=False, batch_size=batch_size)
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_accuracy', patience=self.patience, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
            ]
            
            # Train model
            history = model.fit(
                train_gen,
                steps_per_epoch=train_steps,
                validation_data=val_gen,
                validation_steps=val_steps,
                epochs=self.max_epochs,
                callbacks=callbacks,
                verbose=0
            )
            
            # Get best validation accuracy (proxy for F1 score for speed)
            best_val_acc = max(history.history['val_accuracy'])
            
            # Quick F1 evaluation on a subset for more accurate scoring
            subset_size = min(200, len(self.val_df))
            val_subset = self.val_df.sample(n=subset_size, random_state=42)
            
            predictions = []
            true_labels = []
            
            for _, row in val_subset.iterrows():
                image = self.base_classifier.open_image(row['path'])
                image = tf.expand_dims(image, 0)
                pred = model.predict(image, verbose=0)[0]
                predictions.append(np.argmax(pred))
                true_labels.append(self.base_classifier.label_to_idx[row['label']])
            
            f1_macro = f1_score(true_labels, predictions, average='macro')
            
            # Clean up
            del model
            tf.keras.backend.clear_session()
            
            return f1_macro
            
        except Exception as e:
            print(f"Error in trial: {e}")
            return 0.0  # Return worst possible score on error
    
    def save_trial_result(self, params: Dict[str, Any], score: float, trial_num: int):
        """Save individual trial result"""
        result = {
            'trial': trial_num,
            'score': score,
            'params': params,
            'timestamp': datetime.now().isoformat()
        }
        
        self.search_results.append(result)
        
        # Save to JSON file
        with open(self.run_dir / "logs" / "search_results.json", 'w') as f:
            json.dump(self.search_results, f, indent=2)
        
        # Update best model if needed
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            
            # Save best parameters
            with open(self.run_dir / "best_model" / "best_params.json", 'w') as f:
                json.dump({
                    'score': self.best_score,
                    'params': self.best_params,
                    'trial': trial_num
                }, f, indent=2)
            
            print(f"\n🎉 NEW BEST SCORE: {score:.4f}")
            print(f"Parameters: {params}")
    
    def estimate_remaining_time(self, trial_num: int) -> str:
        """Estimate remaining time based on completed trials"""
        if trial_num == 0:
            return "Calculating..."
        
        if self.search_start_time is None:
            return "Calculating..."
        
        elapsed = time.time() - self.search_start_time
        avg_time_per_trial = elapsed / trial_num
        remaining_trials = self.n_calls - trial_num
        estimated_remaining = remaining_trials * avg_time_per_trial
        
        return str(timedelta(seconds=int(estimated_remaining)))
    
    def objective_function(self, **params):
        """Objective function for Bayesian optimization"""
        self.completed_trials += 1
        
        print(f"\n📊 Trial {self.completed_trials}/{self.n_calls}")
        print(f"⏱️  Estimated remaining: {self.estimate_remaining_time(self.completed_trials)}")
        print(f"🔧 Testing params: {params}")
        
        # Train and evaluate
        start_time = time.time()
        score = self.train_and_evaluate(params)
        trial_time = time.time() - start_time
        
        print(f"⚡ Trial completed in {trial_time:.1f}s - Score: {score:.4f}")
        
        # Save result
        self.save_trial_result(params, score, self.completed_trials)
        
        # Return negative score for minimization
        return -score
    
    def run_search(self):
        """Run the hyperparameter search"""
        print("🚀 Starting Intelligent Hyperparameter Search")
        print(f"📁 Results will be saved to: {self.run_dir}")
        print(f"🎯 Target: {self.n_calls} trials with early stopping")
        
        self.search_start_time = time.time()
        
        # Create objective function with proper decorator
        objective_func = use_named_args(self.search_space)(self.objective_function)
        
        try:
            # Run Bayesian optimization
            result = gp_minimize(
                func=objective_func,
                dimensions=self.search_space,
                n_calls=self.n_calls,
                n_initial_points=5,  # Random exploration first
                acq_func='EI',  # Expected Improvement
                random_state=42
            )
            
            print("\n✅ Hyperparameter search completed!")
            
            # Save final results
            self.save_final_results(result)
            
            # Train best model on full dataset
            self.train_best_model()
            
        except KeyboardInterrupt:
            print("\n⚠️  Search interrupted by user")
            print(f"💾 Saving progress... ({self.completed_trials} trials completed)")
            if self.best_params:
                self.train_best_model()
    
    def save_final_results(self, result):
        """Save final search results and create visualizations"""
        # Save optimization result
        with open(self.run_dir / "logs" / "optimization_result.pkl", 'wb') as f:
            pickle.dump(result, f)
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.search_results)
        results_df.to_csv(self.run_dir / "logs" / "search_results.csv", index=False)
        
        # Plot search progress
        self.plot_search_progress(results_df)
        
        # Analyze parameter importance
        self.analyze_parameter_importance(results_df)
        
        print(f"\n📈 Best score achieved: {self.best_score:.4f}")
        print(f"🏆 Best parameters saved to: {self.run_dir / 'best_model'}")
    
    def plot_search_progress(self, results_df):
        """Plot search progress and parameter relationships"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Search progress
        axes[0, 0].plot(results_df['trial'], results_df['score'], 'b-', alpha=0.7)
        axes[0, 0].plot(results_df['trial'], results_df['score'].cummax(), 'r-', linewidth=2)
        axes[0, 0].set_xlabel('Trial')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_title('Search Progress')
        axes[0, 0].grid(True)
        axes[0, 0].legend(['Trial Score', 'Best Score'])
        
        # Parameter correlation with score
        param_cols = []
        for result in self.search_results:
            for key, value in result['params'].items():
                if isinstance(value, (int, float)):
                    param_cols.append(key)
                    break
        
        if param_cols:
            param_name = param_cols[0] if 'learning_rate' not in param_cols else 'learning_rate'
            if param_name in results_df['params'].iloc[0]:
                param_values = [r['params'][param_name] for r in self.search_results]
                axes[0, 1].scatter(param_values, results_df['score'], alpha=0.6)
                axes[0, 1].set_xlabel(param_name)
                axes[0, 1].set_ylabel('F1 Score')
                axes[0, 1].set_title(f'{param_name} vs Performance')
        
        # Score distribution
        axes[1, 0].hist(results_df['score'], bins=15, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(self.best_score, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('F1 Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Score Distribution')
        
        # Top 10 trials
        top_trials = results_df.nlargest(10, 'score')
        axes[1, 1].barh(range(len(top_trials)), top_trials['score'])
        axes[1, 1].set_xlabel('F1 Score')
        axes[1, 1].set_ylabel('Trial Rank')
        axes[1, 1].set_title('Top 10 Trials')
        
        plt.tight_layout()
        plt.savefig(self.run_dir / "plots" / "search_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_parameter_importance(self, results_df):
        """Analyze which parameters have the most impact"""
        print("\n📊 Parameter Impact Analysis:")
        
        # Group by categorical parameters and calculate mean scores
        categorical_params = ['optimizer', 'fine_tuning']
        
        for param in categorical_params:
            if param in results_df['params'].iloc[0]:
                param_values = [r['params'][param] for r in self.search_results]
                param_df = pd.DataFrame({'param': param_values, 'score': results_df['score']})
                param_means = param_df.groupby('param')['score'].agg(['mean', 'std', 'count'])
                
                print(f"\n{param.upper()}:")
                for idx, row in param_means.iterrows():
                    print(f"  {idx}: {row['mean']:.4f} ± {row['std']:.4f} ({row['count']} trials)")
    
    def train_best_model(self):
        """Train the best model found and save it"""
        if not self.best_params:
            print("❌ No best parameters found")
            return
        
        print(f"\n🏭 Training best model with score {self.best_score:.4f}")
        
        try:
            # Create model with best parameters
            model = self.create_model_from_params(self.best_params)
            
            # Train with more epochs for final model
            batch_size = self.best_params['batch_size']
            train_steps = len(self.train_df) // batch_size
            val_steps = len(self.val_df) // batch_size
            
            train_gen = self.base_classifier.data_generator(self.train_df, augment=True, batch_size=batch_size)
            val_gen = self.base_classifier.data_generator(self.val_df, augment=False, batch_size=batch_size)
            
            callbacks = [
                EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
            ]
            
            # Train for more epochs
            history = model.fit(
                train_gen,
                steps_per_epoch=train_steps,
                validation_data=val_gen,
                validation_steps=val_steps,
                epochs=30,  # More epochs for final model
                callbacks=callbacks,
                verbose=1
            )
            
            # Save the best model
            model.save(self.run_dir / "best_model" / "best_duck_classifier.h5")
            
            # Evaluate on test set
            print("🧪 Evaluating on test set...")
            test_predictions = []
            test_true_labels = []
            
            for _, row in self.test_df.iterrows():
                image = self.base_classifier.open_image(row['path'])
                image = tf.expand_dims(image, 0)
                pred = model.predict(image, verbose=0)[0]
                test_predictions.append(np.argmax(pred))
                test_true_labels.append(self.base_classifier.label_to_idx[row['label']])
            
            test_f1 = f1_score(test_true_labels, test_predictions, average='macro')
            test_accuracy = np.mean(np.array(test_predictions) == np.array(test_true_labels))
            
            # Save final results
            final_results = {
                'best_params': self.best_params,
                'validation_f1': self.best_score,
                'test_f1': test_f1,
                'test_accuracy': test_accuracy,
                'total_trials': self.completed_trials
            }
            
            with open(self.run_dir / "best_model" / "final_results.json", 'w') as f:
                json.dump(final_results, f, indent=2)
            
            print(f"\n🎯 FINAL RESULTS:")
            print(f"   Validation F1: {self.best_score:.4f}")
            print(f"   Test F1: {test_f1:.4f}")
            print(f"   Test Accuracy: {test_accuracy:.4f}")
            print(f"   Model saved to: {self.run_dir / 'best_model'}")
            
        except Exception as e:
            print(f"❌ Error training best model: {e}")


def main():
    """Main function to run hyperparameter optimization"""
    DATA_PATH = "images/Phone"
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data path '{DATA_PATH}' not found!")
        return
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(data_path=DATA_PATH)
    
    # Run search
    optimizer.run_search()


if __name__ == "__main__":
    main() 