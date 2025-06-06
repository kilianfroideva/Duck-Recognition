#!/usr/bin/env python3
"""
Duck Species Classification using Transfer Learning
ARN Project - HEIG-VD

This script implements transfer learning for duck species classification using MobileNetV2.
Classes: Autre, Colvert male, Colvert femelle, Foulque macroule, Grebe huppe
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, 
    RandomFlip, RandomRotation, RandomZoom, RandomBrightness,
    RandomContrast, Resizing, Rescaling
)
from tensorflow.keras import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import PIL.Image
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from matplotlib import cm
import tensorflow.keras.backend as K

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class DuckClassifier:
    def __init__(self, data_path="images/Phone", output_dir="output"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.img_height = 224
        self.img_width = 224
        self.batch_size = 32
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(output_dir) / f"duck_transfer_learning_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.run_dir / "model").mkdir(exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)
        (self.run_dir / "samples").mkdir(exist_ok=True)
        (self.run_dir / "augmentation_examples").mkdir(exist_ok=True)
        (self.run_dir / "grad_cam").mkdir(exist_ok=True)
        
        # Define label mapping
        self.label_names = ["Autre", "Colvert femelle", "Colvert mâle", "Foulque macroule", "Grèbe huppé"]
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_names)}
        
        print(f"Initializing Duck Classifier")
        print(f"Output directory: {self.run_dir}")
        print(f"Classes: {self.label_names}")
    
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
        print("Loading dataset...")
        df = self.create_dataframe_from_directories(self.data_path)
        
        print(f"Total images: {len(df)}")
        print("\nClass distribution:")
        class_counts = df['label'].value_counts()
        print(class_counts)
        
        # Plot class distribution
        plt.figure(figsize=(10, 6))
        sns.barplot(x=class_counts.index, y=class_counts.values)
        plt.title("Class Distribution")
        plt.xlabel("Classes")
        plt.ylabel("Number of Images")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.run_dir / "plots" / "class_distribution.png")
        plt.close()
        
        # Split data: 70% train, 15% val, 15% test
        train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
        
        print(f"\nData splits:")
        print(f"Train: {len(train_df)} images")
        print(f"Validation: {len(val_df)} images")
        print(f"Test: {len(test_df)} images")
        
        return train_df, val_df, test_df
    
    def create_preprocessing_layers(self):
        """Create image preprocessing layers"""
        return Sequential([
            Resizing(self.img_height, self.img_width, crop_to_aspect_ratio=True),
            Rescaling(1.0 / 255.0)
        ])
    
    def create_augmentation_layers(self):
        """Create comprehensive data augmentation using tf.image functions that naturally stay in [0,1] range"""
        def augment_image(image):
            # Start with image in [0,1] range
            img = image
            
            # Geometric augmentations
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img, seed=None)  # Sometimes useful for duck orientations
            
            # Random rotation (using tf.image which handles ranges properly)
            angle = tf.random.uniform([], -0.2, 0.2)  # ±0.2 radians ≈ ±11.5 degrees
            img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))  # 0°, 90°, 180°, 270°
            
            # Photometric augmentations (tf.image functions naturally handle [0,1] range)
            img = tf.image.random_brightness(img, 0.15)  # ±15% brightness, automatically clamped
            img = tf.image.random_contrast(img, 0.8, 1.2)  # Contrast between 0.8x and 1.2x
            img = tf.image.random_saturation(img, 0.8, 1.2)  # Saturation variation
            img = tf.image.random_hue(img, 0.05)  # Small hue shifts
            
            # More advanced augmentations
            # Random jpeg quality (compression artifacts)
            if tf.random.uniform([]) < 0.3:  # 30% chance
                img = tf.image.random_jpeg_quality(img, 75, 95)
            
            # Random crop and resize (zoom effect without border issues)
            if tf.random.uniform([]) < 0.4:  # 40% chance
                img = tf.image.random_crop(img, size=[int(224*0.85), int(224*0.85), 3])
                img = tf.image.resize(img, [224, 224])
            
            # Color channel augmentations
            if tf.random.uniform([]) < 0.2:  # 20% chance
                # Randomly swap color channels
                channels = tf.random.shuffle([0, 1, 2])
                img = tf.gather(img, channels, axis=-1)
            
            # Gaussian noise (very small amount)
            if tf.random.uniform([]) < 0.3:  # 30% chance
                noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.02)
                img = img + noise
            
            # Final safety - tf.clip_by_value to ensure [0,1] range
            # This should rarely be needed with tf.image functions, but ensures absolute safety
            img = tf.clip_by_value(img, 0.0, 1.0)
            
            return img
        
        return augment_image
    
    def open_image(self, path):
        """Open and preprocess a single image"""
        with PIL.Image.open(path) as image:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return self.preprocessing(np.asarray(image))
    
    def open_images(self, paths):
        """Open and preprocess multiple images"""
        return np.stack([self.open_image(path) for path in paths])
    
    def show_augmentation_examples(self, train_df):
        """Show examples of data augmentation"""
        print("Creating augmentation examples...")
        
        # Select one image from each class
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
        plt.savefig(self.run_dir / "augmentation_examples" / "augmentation_examples.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def show_sample_images(self, train_df):
        """Show sample images from each class"""
        print("Creating sample images...")
        
        fig, axes = plt.subplots(len(self.label_names), 5, figsize=(20, 4*len(self.label_names)))
        fig.suptitle("Sample Images from Each Class", fontsize=16)
        
        for class_idx, class_name in enumerate(self.label_names):
            class_images = train_df[train_df['label'] == class_name].sample(min(5, len(train_df[train_df['label'] == class_name])))
            
            for img_idx, (_, row) in enumerate(class_images.iterrows()):
                if img_idx < 5:
                    image = self.open_image(row['path'])
                    axes[class_idx, img_idx].imshow(image)
                    axes[class_idx, img_idx].set_title(f"{class_name}\nSample {img_idx + 1}")
                    axes[class_idx, img_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.run_dir / "samples" / "sample_images.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def balance_dataset(self, train_df):
        """Balance dataset by oversampling minority classes"""
        print("Balancing dataset...")
        
        # Find the maximum class count
        class_counts = train_df['label'].value_counts()
        max_count = class_counts.max()
        
        print(f"Original class distribution:")
        for label, count in class_counts.items():
            print(f"  {label}: {count}")
        
        balanced_dfs = []
        
        # For each class, oversample to match the majority class
        for class_name in self.label_names:
            class_df = train_df[train_df['label'] == class_name]
            current_count = len(class_df)
            
            if current_count < max_count:
                # Calculate how many times to repeat
                repeat_factor = max_count // current_count
                remainder = max_count % current_count
                
                # Repeat the dataframe
                repeated_df = pd.concat([class_df] * repeat_factor, ignore_index=True)
                
                # Add remainder samples
                if remainder > 0:
                    extra_samples = class_df.sample(n=remainder, random_state=42)
                    repeated_df = pd.concat([repeated_df, extra_samples], ignore_index=True)
                
                balanced_dfs.append(repeated_df)
            else:
                balanced_dfs.append(class_df)
        
        # Combine all balanced classes
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        print(f"\nBalanced class distribution:")
        balanced_counts = balanced_df['label'].value_counts()
        for label, count in balanced_counts.items():
            print(f"  {label}: {count}")
        
        return balanced_df
    
    def create_model(self):
        """Create transfer learning model with MobileNetV2"""
        print("Creating model...")
        
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            input_shape=(self.img_height, self.img_width, 3),
            weights='imagenet',
            include_top=False
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom layers
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(len(self.label_names), activation='softmax')
        ])
        
        # Custom F1-macro metric
        def f1_macro(y_true, y_pred):
            y_pred = tf.argmax(y_pred, axis=1)
            y_true = tf.cast(y_true, tf.int64)
            
            # Calculate F1 for each class
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
        
        # Compile model with F1-macro
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy', f1_macro]
        )
        
        return model
    
    def data_generator(self, df, augment=False, batch_size=None):
        """Generate batches of data"""
        if batch_size is None:
            batch_size = self.batch_size
            
        while True:
            # Shuffle data
            df_shuffled = df.sample(frac=1).reset_index(drop=True)
            
            for i in range(0, len(df_shuffled), batch_size):
                batch_df = df_shuffled.iloc[i:i+batch_size]
                
                # Load images
                images = self.open_images(batch_df['path'].values)
                
                # Apply augmentation if specified
                if augment:
                    # Apply augmentation properly to each image
                    augmented_images = []
                    for img in images:
                        # Ensure image is in [0,1] range before augmentation
                        img_aug = self.augmentation(img)
                        augmented_images.append(img_aug)
                    images = tf.stack(augmented_images)
                
                # Convert labels to indices
                labels = np.array([self.label_to_idx[label] for label in batch_df['label']])
                
                yield images, labels
    
    def train_model(self, train_df, val_df):
        """Train the model"""
        print("Training model...")
        
        # Balance the training dataset
        balanced_train_df = self.balance_dataset(train_df)
        
        # Create model
        model = self.create_model()
        
        # Print model summary
        print("\nModel Architecture:")
        model.summary()
        
        # Calculate steps per epoch
        train_steps = len(balanced_train_df) // self.batch_size
        val_steps = len(val_df) // self.batch_size
        
        # Create data generators
        train_gen = self.data_generator(balanced_train_df, augment=True)
        val_gen = self.data_generator(val_df, augment=False)
        
        # Train model
        history = model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            validation_data=val_gen,
            validation_steps=val_steps,
            epochs=20,
            verbose=1
        )
        
        # Save model
        model.save(self.run_dir / "model" / "duck_classifier.h5")
        print(f"Model saved to {self.run_dir / 'model' / 'duck_classifier.h5'}")
        
        # Plot training history
        self.plot_training_history(history)
        
        return model, history
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot F1-macro
        ax2.plot(history.history['f1_macro'], label='Training F1-Macro')
        ax2.plot(history.history['val_f1_macro'], label='Validation F1-Macro')
        ax2.set_title('Model F1-Macro Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1-Macro')
        ax2.legend()
        ax2.grid(True)
        
        # Plot loss
        ax3.plot(history.history['loss'], label='Training Loss')
        ax3.plot(history.history['val_loss'], label='Validation Loss')
        ax3.set_title('Model Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / "plots" / "training_history.png")
        plt.close()
    
    def evaluate_model(self, model, test_df):
        """Evaluate model on test set"""
        print("Evaluating model on test set...")
        
        # Generate predictions
        test_gen = self.data_generator(test_df, augment=False, batch_size=1)
        predictions = []
        true_labels = []
        
        for i, (image, label) in enumerate(test_gen):
            if i >= len(test_df):
                break
            pred = model.predict(image, verbose=0)
            predictions.append(np.argmax(pred))
            true_labels.append(label[0])
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Calculate metrics
        accuracy = np.mean(predictions == true_labels)
        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        f1_macro = f1_score(true_labels, predictions, average='macro')
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Weighted F1-Score: {f1_weighted:.4f}")
        print(f"Macro F1-Score: {f1_macro:.4f}")
        
        # Classification report
        report = classification_report(true_labels, predictions, target_names=self.label_names)
        print("\nClassification Report:")
        print(report)
        
        # Save classification report
        with open(self.run_dir / "classification_report.txt", 'w') as f:
            f.write(f"Test Accuracy: {accuracy:.4f}\n")
            f.write(f"Weighted F1-Score: {f1_weighted:.4f}\n")
            f.write(f"Macro F1-Score: {f1_macro:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(str(report))
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_names, yticklabels=self.label_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(self.run_dir / "plots" / "confusion_matrix.png")
        plt.close()
        
        return predictions, true_labels
    
    def generate_grad_cam(self, model, test_df, predictions, true_labels):
        """Generate Grad-CAM visualizations"""
        print("Generating Grad-CAM visualizations...")
        
        # Create GradCAM++ object
        gradcam = GradcamPlusPlus(
            model,
            model_modifier=ReplaceToLinear(),
            clone=True,
        )
        
        # Select some correctly classified and misclassified examples
        correct_indices = np.where(predictions == true_labels)[0][:5]
        incorrect_indices = np.where(predictions != true_labels)[0][:5]
        
        # Correctly classified examples
        if len(correct_indices) > 0:
            self._plot_grad_cam(gradcam, test_df, correct_indices, predictions, true_labels, "correct")
        
        # Misclassified examples
        if len(incorrect_indices) > 0:
            self._plot_grad_cam(gradcam, test_df, incorrect_indices, predictions, true_labels, "misclassified")
    
    def _plot_grad_cam(self, gradcam, test_df, indices, predictions, true_labels, suffix):
        """Plot Grad-CAM for given indices"""
        fig, axes = plt.subplots(len(indices), 2, figsize=(10, 3*len(indices)))
        if len(indices) == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(indices):
            row = test_df.iloc[idx]
            img = self.open_image(row['path'])
            
            # Generate prediction
            pred = predictions[idx]
            true_label = true_labels[idx]
            
            # Generate CAM
            score = CategoricalScore(pred)
            cam = gradcam(score, img)
            heatmap = np.uint8(cm.get_cmap('jet')(cam[0])[..., :3] * 255)
            
            # Plot original image
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Original\nTrue: {self.label_names[true_label]}\nPred: {self.label_names[pred]}")
            axes[i, 0].axis('off')
            
            # Plot with heatmap overlay
            axes[i, 1].imshow(img)
            axes[i, 1].imshow(heatmap, cmap='jet', alpha=0.5)
            axes[i, 1].set_title(f"Grad-CAM\nTrue: {self.label_names[true_label]}\nPred: {self.label_names[pred]}")
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.run_dir / "grad_cam" / f"grad_cam_{suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_training(self):
        """Run the complete training pipeline"""
        print("="*50)
        print("Duck Species Classification with Transfer Learning")
        print("="*50)
        
        # Load and split data
        train_df, val_df, test_df = self.load_and_split_data()
        
        # Initialize preprocessing and augmentation
        self.preprocessing = self.create_preprocessing_layers()
        self.augmentation = self.create_augmentation_layers()
        
        # Show sample images and augmentation examples
        self.show_sample_images(train_df)
        self.show_augmentation_examples(train_df)
        
        # Train model
        model, history = self.train_model(train_df, val_df)
        
        # Evaluate model
        predictions, true_labels = self.evaluate_model(model, test_df)
        
        # Generate Grad-CAM visualizations
        self.generate_grad_cam(model, test_df, predictions, true_labels)
        
        print(f"\nTraining completed! Results saved to: {self.run_dir}")
        return model, history

if __name__ == "__main__":
    # Initialize and run classifier
    classifier = DuckClassifier()
    model, history = classifier.run_training() 