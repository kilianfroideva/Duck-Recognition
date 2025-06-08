#!/usr/bin/env python3
"""
Complete Duck Species Classification using Transfer Learning
Optimized with best hyperparameters for F1-macro score
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, Resizing, Rescaling, BatchNormalization
)
from tensorflow.keras import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from pathlib import Path
from datetime import datetime
import PIL.Image

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class DuckClassifierOptimized:
    def __init__(self, data_path="images/Phone", output_dir="output", best_params_file="best_model.json"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.img_height = 224
        self.img_width = 224
        
        # Load best hyperparameters
        self.load_best_parameters(best_params_file)
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(output_dir) / f"duck_classification_f1_optimized_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.run_dir / "model").mkdir(exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)
        (self.run_dir / "samples").mkdir(exist_ok=True)
        (self.run_dir / "cv_results").mkdir(exist_ok=True)
        
        # Define label mapping
        self.label_names = ["Autre", "Colvert femelle", "Colvert mâle", "Foulque macroule", "Grèbe huppé"]
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_names)}
        
        # Initialize preprocessing and augmentation
        self.preprocessing = self.create_preprocessing_layers()
        self.augmentation = self.create_augmentation_layers()
    
    def load_best_parameters(self, best_params_file):
        """Load best hyperparameters from JSON file"""
        try:
            with open(best_params_file, 'r') as f:
                best_model_data = json.load(f)
            
            params = best_model_data['params']
            
            # Set hyperparameters
            self.learning_rate = params['learning_rate']
            self.dropout1 = params['dropout1']
            self.dropout2 = params['dropout2']
            self.dense_units = params['dense_units']
            self.use_batch_norm = params['use_batch_norm'] == "True" or params['use_batch_norm'] == True
            self.batch_size = params['batch_size']
            self.optimizer_name = params['optimizer']
            self.fine_tuning = params['fine_tuning']
            self.l2_reg = params['l2_reg']
            self.epochs = 25  # Increased for better training
            
            print(f"Loaded best hyperparameters:")
            print(f"  Learning Rate: {self.learning_rate}")
            print(f"  Dropout1: {self.dropout1}, Dropout2: {self.dropout2}")
            print(f"  Dense Units: {self.dense_units}")
            print(f"  Batch Normalization: {self.use_batch_norm}")
            print(f"  Batch Size: {self.batch_size}")
            print(f"  Optimizer: {self.optimizer_name}")
            print(f"  Fine-tuning: {self.fine_tuning}")
            print(f"  L2 Regularization: {self.l2_reg}")
            
        except FileNotFoundError:
            print(f"Warning: {best_params_file} not found. Using default parameters.")
            self.set_default_parameters()
        except Exception as e:
            print(f"Error loading parameters: {e}. Using default parameters.")
            self.set_default_parameters()
    
    def set_default_parameters(self):
        """Set default hyperparameters if loading fails"""
        self.learning_rate = 0.001
        self.dropout1 = 0.2
        self.dropout2 = 0.3
        self.dense_units = 128
        self.use_batch_norm = False
        self.batch_size = 32
        self.optimizer_name = "adam"
        self.fine_tuning = "frozen"
        self.l2_reg = 1e-4
        self.epochs = 20

    # =================== CREATE DATASET ===================
    def create_dataframe_from_directories(self, path):
        """Create DataFrame from directory structure"""
        data = []
        for label_dir in os.listdir(path):
            label_path = os.path.join(path, label_dir)
            if not os.path.isdir(label_path):
                continue
            
            for file in os.listdir(label_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    data.append([os.path.join(label_path, file), label_dir])
        
        return pd.DataFrame(data, columns=["path", "label"])
    
    def load_and_split_data(self):
        """Load data and create train/validation/test splits"""
        df = self.create_dataframe_from_directories(self.data_path)
        
        # Split data: 70% train, 15% val, 15% test
        train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
        
        return train_df, val_df, test_df
    
    # =================== EXPLORE DATASET ===================
    def explore_dataset(self, train_df, val_df, test_df):
        """Count examples per class and plot histogram"""
        # Plot original class distribution
        class_counts = train_df['label'].value_counts()
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        sns.barplot(x=class_counts.index, y=class_counts.values)
        plt.title("Training Set - Class Distribution")
        plt.xlabel("Classes")
        plt.ylabel("Number of Images")
        plt.xticks(rotation=45, ha='right')
        
        # Plot distribution comparison
        splits_data = []
        for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            counts = df['label'].value_counts()
            for label in self.label_names:
                splits_data.append({
                    'Split': name,
                    'Class': label,
                    'Count': counts.get(label, 0)
                })
        
        splits_df = pd.DataFrame(splits_data)
        
        plt.subplot(2, 2, 2)
        sns.barplot(data=splits_df, x='Class', y='Count', hue='Split')
        plt.title("Distribution Comparison Across Splits")
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        
        # Show sample images
        plt.subplot(2, 1, 2)
        self.show_sample_images(train_df)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / "plots" / "dataset_exploration.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return class_counts
    
    def show_sample_images(self, train_df):
        """Display sample images from dataset"""
        fig, axes = plt.subplots(1, len(self.label_names), figsize=(20, 4))
        fig.suptitle("Sample Images from Each Class", fontsize=16)
        
        for class_idx, class_name in enumerate(self.label_names):
            class_images = train_df[train_df['label'] == class_name]
            if len(class_images) > 0:
                sample_path = class_images.iloc[0]['path']
                image = self.open_image(sample_path)
                axes[class_idx].imshow(image)
                axes[class_idx].set_title(class_name)
                axes[class_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.run_dir / "samples" / "sample_images.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # =================== PREPROCESSING STEPS ===================
    def create_preprocessing_layers(self):
        """Create image preprocessing layers"""
        return Sequential([
            Resizing(self.img_height, self.img_width, crop_to_aspect_ratio=True),
            Rescaling(1.0 / 255.0)
        ])
    
    def create_augmentation_layers(self):
        """Create data augmentation layers"""
        def augment_image(image):
            img = image
            
            # Geometric augmentations
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
            
            # Photometric augmentations
            img = tf.image.random_brightness(img, 0.15)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.image.random_saturation(img, 0.8, 1.2)
            img = tf.image.random_hue(img, 0.05)

            
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img
        
        return augment_image
    
    def show_augmentation_examples(self, train_df):
        """Plot examples of preprocessed and augmented images"""
        fig, axes = plt.subplots(len(self.label_names), 5, figsize=(20, 4*len(self.label_names)))
        fig.suptitle("Data Augmentation Examples", fontsize=16)
        
        for class_idx, class_name in enumerate(self.label_names):
            class_images = train_df[train_df['label'] == class_name]
            if len(class_images) > 0:
                sample_path = class_images.iloc[0]['path']
                original_image = self.open_image(sample_path)
                
                # Show original
                axes[class_idx, 0].imshow(original_image)
                axes[class_idx, 0].set_title(f"{class_name}\n(Original)")
                axes[class_idx, 0].axis('off')
                
                # Show 4 augmented versions
                for aug_idx in range(1, 5):
                    augmented = self.augmentation(original_image)
                    axes[class_idx, aug_idx].imshow(augmented)
                    axes[class_idx, aug_idx].set_title(f"Augmented {aug_idx}")
                    axes[class_idx, aug_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.run_dir / "plots" / "augmentation_examples.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def open_image(self, path):
        """Open and preprocess a single image"""
        with PIL.Image.open(path) as image:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return self.preprocessing(np.asarray(image))
    
    def open_images(self, paths):
        """Open and preprocess multiple images"""
        return np.stack([self.open_image(path) for path in paths])
    
    # =================== DEFINE MODEL ===================
    def f1_macro(self, y_true, y_pred):
        """Custom F1-macro metric for multi-class classification"""
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.cast(y_true, tf.int64)
        
        f1_scores = []
        for i in range(len(self.label_names)):
            tp = tf.reduce_sum(tf.cast((y_true == i) & (y_pred == i), tf.float32))
            fp = tf.reduce_sum(tf.cast((y_true != i) & (y_pred == i), tf.float32))
            fn = tf.reduce_sum(tf.cast((y_true == i) & (y_pred != i), tf.float32))
            
            precision = tp / (tp + fp + K.epsilon())
            recall = tp / (tp + fn + K.epsilon())
            f1 = 2 * precision * recall / (precision + recall + K.epsilon())
            f1_scores.append(f1)
        
        return tf.reduce_mean(f1_scores)
    
    def get_model(self):
        """Create transfer learning model with optimized hyperparameters"""
        base_model = MobileNetV2(
            input_shape=(224, 224, 3),
            weights='imagenet',
            include_top=False
        )
        
        # Set fine-tuning strategy
        if self.fine_tuning == 'frozen':
            base_model.trainable = False
        elif self.fine_tuning == 'partial':
            base_model.trainable = True
            # Freeze first 80% of layers
            freeze_until = int(len(base_model.layers) * 0.8)
            for layer in base_model.layers[:freeze_until]:
                layer.trainable = False
        else:  # full fine-tuning
            base_model.trainable = True
        
        # Build model with optimized architecture
        layers = [
            base_model,
            GlobalAveragePooling2D(),
            Dropout(self.dropout1)
        ]
        
        # Add dense layer with L2 regularization
        layers.append(Dense(
            self.dense_units, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        ))
        
        # Add batch normalization if enabled
        if self.use_batch_norm:
            layers.append(BatchNormalization())
        
        layers.extend([
            Dropout(self.dropout2),
            Dense(len(self.label_names), activation='softmax')
        ])
        
        model = Sequential(layers)
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy', self.f1_macro]
        )
        
        return model
    
    # =================== MODEL TRAINING WITH CROSS-VALIDATION ===================
    def calculate_class_weights(self, train_df):
        """Calculate class weights for imbalanced dataset"""
        class_counts = train_df['label'].value_counts()
        total_samples = len(train_df)
        n_classes = len(self.label_names)
        
        class_weights = {}
        for class_name in self.label_names:
            class_idx = self.label_to_idx[class_name]
            if class_name in class_counts:
                class_count = class_counts[class_name]
                weight = total_samples / (n_classes * class_count)
            else:
                weight = 1.0
            class_weights[class_idx] = weight
        
        # Plot class weights
        plt.figure(figsize=(10, 6))
        weights_df = pd.DataFrame([
            {'Class': class_name, 'Weight': class_weights[self.label_to_idx[class_name]]}
            for class_name in self.label_names
        ])
        sns.barplot(data=weights_df, x='Class', y='Weight')
        plt.title("Class Weights for Imbalance Handling")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.run_dir / "plots" / "class_weights.png")
        plt.close()
        
        return class_weights
    
    def data_generator(self, df, augment=False, batch_size=None):
        """Generate batches of data"""
        if batch_size is None:
            batch_size = self.batch_size
            
        while True:
            df_shuffled = df.sample(frac=1).reset_index(drop=True)
            
            for i in range(0, len(df_shuffled), batch_size):
                batch_df = df_shuffled.iloc[i:i+batch_size]
                
                images = self.open_images(batch_df['path'].values)
                
                if augment:
                    augmented_images = []
                    for img in images:
                        img_aug = self.augmentation(img)
                        augmented_images.append(img_aug)
                    images = tf.stack(augmented_images)
                
                labels = np.array([self.label_to_idx[label] for label in batch_df['label']])
                
                yield images, labels
    
    def data_generator_with_weights(self, df, class_weights, augment=False, batch_size=None):
        """Generate batches of data with sample weights"""
        if batch_size is None:
            batch_size = self.batch_size
            
        while True:
            df_shuffled = df.sample(frac=1).reset_index(drop=True)
            
            for i in range(0, len(df_shuffled), batch_size):
                batch_df = df_shuffled.iloc[i:i+batch_size]
                
                images = self.open_images(batch_df['path'].values)
                
                if augment:
                    augmented_images = []
                    for img in images:
                        img_aug = self.augmentation(img)
                        augmented_images.append(img_aug)
                    images = tf.stack(augmented_images)
                
                labels = np.array([self.label_to_idx[label] for label in batch_df['label']])
                sample_weights = np.array([class_weights[label] for label in labels])
                
                yield images, labels, sample_weights
    
    def cross_validate_model(self, train_df, n_folds=5):
        """Perform cross-validation on the training data"""
        print(f"\nStarting {n_folds}-fold cross-validation...")
        
        # Combine data for cross-validation
        cv_results = []
        fold_histories = []
        
        # Create stratified k-fold
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Convert to indices for splitting
        train_indices = np.arange(len(train_df))
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_indices)):
            print(f"\nTraining fold {fold + 1}/{n_folds}...")
            
            # Split data
            fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
            fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)
            
            # Calculate class weights for this fold
            class_weights = self.calculate_class_weights(fold_train_df)
            
            # Create model
            model = self.get_model()
            
            # Prepare generators
            train_steps = len(fold_train_df) // self.batch_size
            val_steps = len(fold_val_df) // self.batch_size
            
            train_gen = self.data_generator_with_weights(fold_train_df, class_weights, augment=True)
            val_gen = self.data_generator(fold_val_df, augment=False)
            
            # Define callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_f1_macro',
                    patience=5,
                    restore_best_weights=True,
                    mode='max',
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Train model
            history = model.fit(
                train_gen,
                steps_per_epoch=train_steps,
                validation_data=val_gen,
                validation_steps=val_steps,
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # Get best scores
            best_val_acc = max(history.history['val_accuracy'])
            best_val_f1 = max(history.history['val_f1_macro'])
            
            cv_results.append({
                'fold': fold + 1,
                'val_accuracy': best_val_acc,
                'val_f1_macro': best_val_f1
            })
            
            fold_histories.append(history)
            
            # Save fold model
            fold_model_path = self.run_dir / "cv_results" / f"fold_{fold+1}_model.h5"
            model.save(str(fold_model_path))
        
        # Calculate cross-validation statistics
        cv_df = pd.DataFrame(cv_results)
        cv_stats = {
            'mean_accuracy': cv_df['val_accuracy'].mean(),
            'std_accuracy': cv_df['val_accuracy'].std(),
            'mean_f1_macro': cv_df['val_f1_macro'].mean(),
            'std_f1_macro': cv_df['val_f1_macro'].std()
        }
        
        # Save CV results
        cv_df.to_csv(self.run_dir / "cv_results" / "cv_results.csv", index=False)
        
        with open(self.run_dir / "cv_results" / "cv_stats.json", 'w') as f:
            json.dump(cv_stats, f, indent=2)
        
        # Plot CV results
        self.plot_cv_results(cv_df, fold_histories)
        
        print(f"\nCross-validation completed!")
        print(f"Mean Accuracy: {cv_stats['mean_accuracy']:.4f} ± {cv_stats['std_accuracy']:.4f}")
        print(f"Mean F1-Macro: {cv_stats['mean_f1_macro']:.4f} ± {cv_stats['std_f1_macro']:.4f}")
        
        return cv_stats, fold_histories
    
    def plot_cv_results(self, cv_df, fold_histories):
        """Plot cross-validation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Cross-Validation Results", fontsize=16)
        
        # CV scores per fold
        axes[0, 0].bar(cv_df['fold'], cv_df['val_accuracy'], alpha=0.7, label='Accuracy')
        axes[0, 0].bar(cv_df['fold'], cv_df['val_f1_macro'], alpha=0.7, label='F1-Macro')
        axes[0, 0].set_title('Validation Scores by Fold')
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Training curves for all folds - Accuracy
        for i, history in enumerate(fold_histories):
            axes[0, 1].plot(history.history['val_accuracy'], alpha=0.6, label=f'Fold {i+1}')
        axes[0, 1].set_title('Validation Accuracy Across Folds')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training curves for all folds - F1
        for i, history in enumerate(fold_histories):
            axes[1, 0].plot(history.history['val_f1_macro'], alpha=0.6, label=f'Fold {i+1}')
        axes[1, 0].set_title('Validation F1-Macro Across Folds')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1-Macro')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Statistical summary
        axes[1, 1].text(0.1, 0.8, f"Mean Accuracy: {cv_df['val_accuracy'].mean():.4f} ± {cv_df['val_accuracy'].std():.4f}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.6, f"Mean F1-Macro: {cv_df['val_f1_macro'].mean():.4f} ± {cv_df['val_f1_macro'].std():.4f}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.4, f"Best Fold (F1): {cv_df.loc[cv_df['val_f1_macro'].idxmax(), 'fold']}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.2, f"Best F1 Score: {cv_df['val_f1_macro'].max():.4f}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Cross-Validation Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.run_dir / "plots" / "cross_validation_results.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_final_model(self, train_df, val_df):
        """Train final model on full training data"""
        print("\nTraining final model on full training data...")
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(train_df)
        
        # Create model
        model = self.get_model()
        
        # Calculate steps per epoch
        train_steps = len(train_df) // self.batch_size
        val_steps = len(val_df) // self.batch_size
        
        # Define callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=str(self.run_dir / "model" / "best_duck_classifier_f1_optimized.h5"),
                monitor='val_f1_macro',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_f1_macro',
                patience=7,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Create generators
        train_gen = self.data_generator_with_weights(train_df, class_weights, augment=True)
        val_gen = self.data_generator(val_df, augment=False)
        
        # Train model
        history = model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            validation_data=val_gen,
            validation_steps=val_steps,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best model
        best_model = load_model(
            str(self.run_dir / "model" / "best_duck_classifier_f1_optimized.h5"),
            custom_objects={'f1_macro': self.f1_macro}
        )
        
        return best_model, history
    
    # =================== RESULTS ===================
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Final Model Training History", fontsize=16)
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training')
        axes[0, 1].plot(history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(history.history['f1_macro'], label='Training')
        axes[1, 0].plot(history.history['val_f1_macro'], label='Validation')
        axes[1, 0].set_title('F1-Macro Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate (if available)
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.run_dir / "plots" / "final_training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # =================== EVALUATE MODEL ON TEST SET ===================
    def evaluate_model(self, model, test_df):
        """Evaluate model on test set"""
        # Generate predictions
        predictions = []
        true_labels = []
        prediction_probabilities = []
        
        for _, row in test_df.iterrows():
            image = self.open_image(row['path'])
            image = tf.expand_dims(image, 0)
            
            pred_probs = model.predict(image, verbose=0)[0]
            predictions.append(np.argmax(pred_probs))
            true_labels.append(self.label_to_idx[row['label']])
            prediction_probabilities.append(pred_probs)
        
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        prediction_probabilities = np.array(prediction_probabilities)
        
        # Calculate metrics
        accuracy = np.mean(predictions == true_labels)
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        
        # Generate classification report
        report = classification_report(
            true_labels, 
            predictions, 
            target_names=self.label_names,
            output_dict=True
        )
        
        # Generate classification report text
        report_text = classification_report(
            true_labels, 
            predictions, 
            target_names=self.label_names
        )
        
        # Save classification report
        with open(self.run_dir / "classification_report.txt", 'w') as f:
            f.write("DUCK SPECIES CLASSIFICATION - TEST SET EVALUATION\n")
            f.write("F1-MACRO OPTIMIZED MODEL\n")
            f.write("="*60 + "\n\n")
            f.write(f"Test Accuracy: {accuracy:.4f}\n")
            f.write(f"F1-Macro Score: {f1_macro:.4f}\n")
            f.write(f"F1-Weighted Score: {f1_weighted:.4f}\n\n")
            f.write("Best Hyperparameters Used:\n")
            f.write(f"  Learning Rate: {self.learning_rate}\n")
            f.write(f"  Dropout1: {self.dropout1}, Dropout2: {self.dropout2}\n")
            f.write(f"  Dense Units: {self.dense_units}\n")
            f.write(f"  Batch Normalization: {self.use_batch_norm}\n")
            f.write(f"  Batch Size: {self.batch_size}\n")
            f.write(f"  Fine-tuning Strategy: {self.fine_tuning}\n")
            f.write(f"  L2 Regularization: {self.l2_reg}\n\n")
            f.write("Detailed Classification Report:\n")
            f.write("-" * 40 + "\n")
            f.write(report_text)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(true_labels, predictions)
        
        # Show most confident predictions
        self.show_confident_predictions(test_df, predictions, true_labels, prediction_probabilities)
        
        return predictions, true_labels, prediction_probabilities, report
    
    def plot_confusion_matrix(self, true_labels, predictions):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.label_names,
            yticklabels=self.label_names
        )
        plt.title("Confusion Matrix - F1-Optimized Model")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.run_dir / "plots" / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # =================== DISPLAY MISCLASSIFIED IMAGES ===================
    def show_confident_predictions(self, test_df, predictions, true_labels, prediction_probabilities):
        """Display most confident correct and incorrect predictions"""
        # Calculate confidence scores
        confidence_scores = np.max(prediction_probabilities, axis=1)
        
        # Find most confident correct predictions
        correct_mask = predictions == true_labels
        if np.any(correct_mask):
            correct_confidences = confidence_scores[correct_mask]
            correct_indices = np.where(correct_mask)[0]
            top_correct_idx = correct_indices[np.argsort(correct_confidences)[-5:]][::-1]
            
            self.plot_prediction_examples(
                test_df, predictions, true_labels, prediction_probabilities,
                top_correct_idx, "Most Confident Correct Predictions", "correct"
            )
        
        # Find most confident incorrect predictions
        incorrect_mask = predictions != true_labels
        if np.any(incorrect_mask):
            incorrect_confidences = confidence_scores[incorrect_mask]
            incorrect_indices = np.where(incorrect_mask)[0]
            top_incorrect_idx = incorrect_indices[np.argsort(incorrect_confidences)[-5:]][::-1]
            
            self.plot_prediction_examples(
                test_df, predictions, true_labels, prediction_probabilities,
                top_incorrect_idx, "Most Confident Incorrect Predictions", "incorrect"
            )
    
    def plot_prediction_examples(self, test_df, predictions, true_labels, prediction_probabilities, 
                               indices, title, filename):
        """Plot prediction examples"""
        fig, axes = plt.subplots(1, min(5, len(indices)), figsize=(20, 4))
        fig.suptitle(title, fontsize=16)
        
        if len(indices) == 1:
            axes = [axes]
        
        for i, idx in enumerate(indices[:5]):
            if i >= len(axes):
                break
                
            row = test_df.iloc[idx]
            image = self.open_image(row['path'])
            
            pred_label = self.label_names[predictions[idx]]
            true_label = self.label_names[true_labels[idx]]
            confidence = np.max(prediction_probabilities[idx])
            
            axes[i].imshow(image)
            axes[i].set_title(
                f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}"
            )
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(indices), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.run_dir / "plots" / f"most_confident_{filename}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # =================== MAIN PIPELINE ===================
    def run_complete_pipeline(self):
        """Run the complete classification pipeline with cross-validation"""
        print("Starting F1-Macro Optimized Duck Species Classification Pipeline")
        print(f"Output directory: {self.run_dir}")
        
        # 1. Create Dataset
        print("\n1. Loading and splitting dataset...")
        train_df, val_df, test_df = self.load_and_split_data()
        
        # 2. Explore Dataset
        print("\n2. Exploring dataset...")
        class_counts = self.explore_dataset(train_df, val_df, test_df)
        
        # 3. Preprocessing Steps - Show augmentation examples
        print("\n3. Creating augmentation examples...")
        self.show_augmentation_examples(train_df)
        
        # 4. Cross-validation
        print("\n4. Performing cross-validation...")
        cv_stats, fold_histories = self.cross_validate_model(train_df)
        
        # 5. Train final model
        print("\n5. Training final model...")
        model, history = self.train_final_model(train_df, val_df)
        
        # 6. Plot training history
        print("\n6. Plotting training results...")
        self.plot_training_history(history)
        
        # 7. Evaluate on test set
        print("\n7. Evaluating on test set...")
        predictions, true_labels, pred_probs, report = self.evaluate_model(model, test_df)
        
        # Print summary
        accuracy = np.mean(predictions == true_labels)
        f1_macro = f1_score(true_labels, predictions, average='macro')
        
        print(f"\nPipeline completed successfully!")
        print(f"Cross-validation F1-Macro: {cv_stats['mean_f1_macro']:.4f} ± {cv_stats['std_f1_macro']:.4f}")
        print(f"Final Test Accuracy: {accuracy:.4f}")
        print(f"Final Test F1-Macro Score: {f1_macro:.4f}")
        print(f"Results saved to: {self.run_dir}")
        
        return model, history, predictions, true_labels, cv_stats


def main():
    """Main function to run the optimized duck classification pipeline"""
    DATA_PATH = "images/Phone"  # Update this path as needed
    OUTPUT_DIR = "output"
    BEST_PARAMS_FILE = "best_model.json"
    
    # Check if data path exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data path '{DATA_PATH}' does not exist!")
        print("Please update the DATA_PATH variable to point to your duck images directory.")
        return
    
    # Initialize classifier
    classifier = DuckClassifierOptimized(
        data_path=DATA_PATH, 
        output_dir=OUTPUT_DIR,
        best_params_file=BEST_PARAMS_FILE
    )
    
    # Run complete pipeline
    model, history, predictions, true_labels, cv_stats = classifier.run_complete_pipeline()


if __name__ == "__main__":
    main() 