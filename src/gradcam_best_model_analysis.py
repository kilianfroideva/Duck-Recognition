#!/usr/bin/env python3
"""
Grad-CAM Analysis for Best Model: Most Confident Correct and Incorrect Predictions
Analyzes the first 10 most confident correct and incorrect predictions for each class
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import PIL.Image
from datetime import datetime
from matplotlib.colors import ListedColormap

# Grad-CAM specific imports
try:
    from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
    from tf_keras_vis.utils.scores import CategoricalScore
    print("✅ tf-keras-vis imported successfully")
except ImportError:
    print("❌ tf-keras-vis not found. Install with: pip install tf-keras-vis")
    exit(1)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class GradCAMBestModelAnalyzer:
    def __init__(self, 
                 best_model_path="report/images/best_model_duck_classification_f1_optimized_20250608_195712/model/best_duck_classifier_f1_optimized.h5",
                 best_params_file="best_model.json",
                 data_path="images/Phone",
                 output_base_dir="output"):
        
        self.best_model_path = best_model_path
        self.data_path = data_path
        self.img_height = 224
        self.img_width = 224
        
        # Create output directory within the specified base directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_base_dir) / f"gradcam_best_model_analysis_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "confident_correct").mkdir(exist_ok=True)
        (self.output_dir / "confident_incorrect").mkdir(exist_ok=True)
        (self.output_dir / "class_analysis").mkdir(exist_ok=True)
        
        # Define label mapping
        self.label_names = ["Autre", "Colvert femelle", "Colvert mâle", "Foulque macroule", "Grèbe huppé"]
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_names)}
        
        # Load best parameters
        self.load_best_parameters(best_params_file)
        
        print(f"🎯 GradCAM Best Model Analyzer initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🏆 Model path: {self.best_model_path}")
    
    def load_best_parameters(self, best_params_file):
        """Load best hyperparameters from JSON file"""
        try:
            with open(best_params_file, 'r') as f:
                best_model_data = json.load(f)
            
            self.best_params = best_model_data['params']
            self.best_score = best_model_data['score']
            
            print(f"📊 Best F1-score: {self.best_score:.4f}")
            print(f"⚙️  Best parameters loaded: {self.best_params}")
            
        except FileNotFoundError:
            print(f"⚠️  Warning: {best_params_file} not found.")
            self.best_params = {}
            self.best_score = 0.0
    
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
    
    def open_image(self, path):
        """Open and preprocess image"""
        try:
            with PIL.Image.open(path) as img:
                img = img.convert('RGB')
                img = img.resize((self.img_width, self.img_height))
                img_array = np.array(img) / 255.0
                return img_array
        except Exception as e:
            print(f"Error opening image {path}: {e}")
            return np.zeros((self.img_height, self.img_width, 3))
    
    def load_best_model(self):
        """Load the best trained model"""
        print(f"🔄 Loading best model from {self.best_model_path}")
        try:
            # Load the model
            self.best_model = load_model(self.best_model_path, custom_objects={"f1_macro": self.f1_macro})
            print("✅ Best model loaded successfully")
            
            # Create a Grad-CAM compatible model
            self.gradcam_model = self.create_gradcam_model()
            print("✅ Grad-CAM model created successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def f1_macro(self, y_true, y_pred):
        """F1-macro metric for model loading"""
        return tf.py_function(self._f1_macro_np, [y_true, y_pred], tf.float32)
    
    def _f1_macro_np(self, y_true, y_pred):
        """Numpy implementation of F1-macro"""
        from sklearn.metrics import f1_score
        y_true_np = y_true.numpy()
        y_pred_np = np.argmax(y_pred.numpy(), axis=1)
        return f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    
    def create_gradcam_model(self):
        """Create a model compatible with Grad-CAM"""
        # Create the same architecture as the best model
        base_model = MobileNetV2(
            input_shape=(224, 224, 3),
            weights='imagenet',
            include_top=False
        )
        
        # Set fine-tuning strategy
        fine_tuning = self.best_params.get('fine_tuning', 'frozen')
        if fine_tuning == 'frozen':
            base_model.trainable = False
        elif fine_tuning == 'partial':
            base_model.trainable = True
            freeze_until = int(len(base_model.layers) * 0.8)
            for layer in base_model.layers[:freeze_until]:
                layer.trainable = False
        else:  # full
            base_model.trainable = True
        
        # Build the same architecture
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.best_params.get('dropout1', 0.1))(x)
        x = Dense(self.best_params.get('dense_units', 512), activation='relu')(x)
        
        if self.best_params.get('use_batch_norm', True):
            x = BatchNormalization()(x)
        
        x = Dropout(self.best_params.get('dropout2', 0.35))(x)
        predictions = Dense(len(self.label_names), activation='softmax')(x)
        
        gradcam_model = Model(inputs=base_model.input, outputs=predictions)
        
        # Transfer weights from the best model
        gradcam_model.set_weights(self.best_model.get_weights())
        
        # Compile with standard metrics
        gradcam_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return gradcam_model
    
    def load_and_evaluate_test_data(self):
        """Load test data and get predictions"""
        print("🔄 Loading and evaluating test data...")
        
        # Load all data and split
        df = self.create_dataframe_from_directories(self.data_path)
        from sklearn.model_selection import train_test_split
        
        # Use the same split as in training (70% train, 15% val, 15% test)
        train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
        
        # Load test images and labels
        test_images = []
        test_labels = []
        test_paths = []
        
        for _, row in test_df.iterrows():
            img = self.open_image(row['path'])
            test_images.append(img)
            test_labels.append(self.label_to_idx[row['label']])
            test_paths.append(row['path'])
        
        self.test_images = np.array(test_images)
        self.test_labels = np.array(test_labels)
        self.test_paths = test_paths
        
        print(f"📊 Test set size: {len(self.test_images)}")
        
        # Get predictions
        print("🔄 Getting predictions...")
        predictions_proba = self.gradcam_model.predict(self.test_images)
        self.predictions = np.argmax(predictions_proba, axis=1)
        self.prediction_probabilities = np.max(predictions_proba, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(self.predictions == self.test_labels)
        print(f"📈 Test accuracy: {accuracy:.4f}")
        
        return test_df
    
    def analyze_confident_predictions(self, n_per_class=10):
        """Analyze most confident correct and incorrect predictions for each class"""
        print(f"🔍 Analyzing top {n_per_class} confident predictions per class...")
        
        self.confident_correct = {}
        self.confident_incorrect = {}
        
        for class_idx, class_name in enumerate(self.label_names):
            # Find indices for this class
            class_indices = np.where(self.test_labels == class_idx)[0]
            
            if len(class_indices) == 0:
                print(f"⚠️  No test samples found for class {class_name}")
                continue
            
            # Get predictions and confidences for this class
            class_predictions = self.predictions[class_indices]
            class_confidences = self.prediction_probabilities[class_indices]
            
            # Correct predictions
            correct_mask = class_predictions == class_idx
            correct_indices = class_indices[correct_mask]
            correct_confidences = class_confidences[correct_mask]
            
            # Incorrect predictions
            incorrect_mask = class_predictions != class_idx
            incorrect_indices = class_indices[incorrect_mask]
            incorrect_confidences = class_confidences[incorrect_mask]
            
            # Sort by confidence and get top N
            if len(correct_indices) > 0:
                sorted_correct = correct_indices[np.argsort(correct_confidences)[::-1]]
                self.confident_correct[class_name] = sorted_correct[:min(n_per_class, len(sorted_correct))]
            else:
                self.confident_correct[class_name] = []
            
            if len(incorrect_indices) > 0:
                sorted_incorrect = incorrect_indices[np.argsort(incorrect_confidences)[::-1]]
                self.confident_incorrect[class_name] = sorted_incorrect[:min(n_per_class, len(sorted_incorrect))]
            else:
                self.confident_incorrect[class_name] = []
            
            print(f"📊 {class_name}: {len(self.confident_correct[class_name])} confident correct, {len(self.confident_incorrect[class_name])} confident incorrect")
    
    def create_gradcam_heatmap(self, img_idx):
        """Create Grad-CAM heatmap for a specific image"""
        img = self.test_images[img_idx]
        predicted_class = self.predictions[img_idx]
        
        # Create score function for the predicted class
        score = CategoricalScore(predicted_class)
        
        # Generate Grad-CAM
        gradcam = GradcamPlusPlus(
            self.gradcam_model,
            model_modifier=ReplaceToLinear(),
            clone=False
        )
        
        cam = gradcam(score, img)
        heatmap = np.uint8(cm.get_cmap('jet')(cam[0])[..., :3] * 255)
        
        return heatmap
    
    def visualize_class_analysis(self, class_name, prediction_type="correct"):
        """Create detailed Grad-CAM visualization for a specific class"""
        if prediction_type == "correct":
            indices = self.confident_correct[class_name]
            title = f"Most Confident Correct Predictions - {class_name}"
            color = 'green'
        else:
            indices = self.confident_incorrect[class_name]
            title = f"Most Confident Incorrect Predictions - {class_name}"
            color = 'red'
        
        if len(indices) == 0:
            print(f"⚠️  No {prediction_type} predictions for {class_name}")
            return
        
        n_images = len(indices)
        cols = min(5, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 4, rows * 8))
        
        # Handle single image case
        if n_images == 1:
            axes = axes.reshape(2, 1)
        elif rows == 1:
            axes = axes.reshape(2, -1)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i, idx in enumerate(indices):
            row = (i // cols) * 2
            col = i % cols
            
            # Get image information
            img = self.test_images[idx]
            true_label = self.label_names[self.test_labels[idx]]
            pred_label = self.label_names[self.predictions[idx]]
            confidence = self.prediction_probabilities[idx]
            
            # Generate Grad-CAM heatmap
            try:
                heatmap = self.create_gradcam_heatmap(idx)
            except Exception as e:
                print(f"Error creating heatmap for image {idx}: {e}")
                heatmap = np.zeros_like(img)
            
            # Original image
            axes[row, col].imshow(img, vmin=0, vmax=1)
            axes[row, col].set_title(f"Original\nTrue: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}", fontsize=10)
            axes[row, col].axis('off')
            
            # Grad-CAM overlay
            axes[row + 1, col].imshow(img, vmin=0, vmax=1)
            axes[row + 1, col].imshow(heatmap, cmap='jet', alpha=0.5)
            
            status = 'CORRECT' if prediction_type == "correct" else 'INCORRECT'
            axes[row + 1, col].set_title(f"Grad-CAM Overlay\n{status}", fontsize=10, color=color, fontweight='bold')
            axes[row + 1, col].axis('off')
        
        # Hide empty subplots
        for i in range(len(indices), rows * cols):
            row = (i // cols) * 2
            col = i % cols
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].axis('off')
                axes[row + 1, col].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{class_name.replace(' ', '_')}_{prediction_type}.png"
        save_path = self.output_dir / "class_analysis" / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved {title} to {save_path}")
    
    def create_summary_visualization(self):
        """Create a summary visualization showing overview of confident predictions"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Confident correct predictions summary (one per class)
        ax1.set_title("Most Confident Correct Predictions (Best per Class)", fontsize=14, fontweight='bold', color='green')
        for class_idx, class_name in enumerate(self.label_names):
            if len(self.confident_correct[class_name]) > 0:
                idx = self.confident_correct[class_name][0]  # Most confident
                img = self.test_images[idx]
                confidence = self.prediction_probabilities[idx]
                
                # Create subplot
                ax = fig.add_subplot(2, 2, 1)
                if class_idx == 0:
                    ax.imshow(img)
                    ax.text(0.02, 0.98, f"{class_name}\nConf: {confidence:.3f}", 
                           transform=ax.transAxes, fontsize=12, fontweight='bold',
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    ax.axis('off')
        
        # 2. Confident incorrect predictions summary
        ax2.set_title("Most Confident Incorrect Predictions (Worst per Class)", fontsize=14, fontweight='bold', color='red')
        
        # 3. Class distribution in confident predictions
        correct_counts = [len(self.confident_correct[name]) for name in self.label_names]
        incorrect_counts = [len(self.confident_incorrect[name]) for name in self.label_names]
        
        x = np.arange(len(self.label_names))
        width = 0.35
        
        ax3.bar(x - width/2, correct_counts, width, label='Confident Correct', color='green', alpha=0.7)
        ax3.bar(x + width/2, incorrect_counts, width, label='Confident Incorrect', color='red', alpha=0.7)
        ax3.set_xlabel('Classes')
        ax3.set_ylabel('Number of Predictions')
        ax3.set_title('Confident Predictions Distribution by Class')
        ax3.set_xticks(x)
        ax3.set_xticklabels(self.label_names, rotation=45, ha='right')
        ax3.legend()
        
        # 4. Confidence distribution
        all_confidences = self.prediction_probabilities
        correct_confidences = all_confidences[self.predictions == self.test_labels]
        incorrect_confidences = all_confidences[self.predictions != self.test_labels]
        
        ax4.hist(correct_confidences, bins=20, alpha=0.7, label='Correct', color='green')
        ax4.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect', color='red')
        ax4.set_xlabel('Confidence Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Confidence Distribution: Correct vs Incorrect')
        ax4.legend()
        
        plt.tight_layout()
        summary_path = self.output_dir / "summary_analysis.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Summary analysis saved to {summary_path}")
    
    def run_complete_analysis(self):
        """Run the complete Grad-CAM analysis"""
        print("🚀 Starting complete Grad-CAM analysis for best model...")
        
        # Load model and data
        self.load_best_model()
        test_df = self.load_and_evaluate_test_data()
        
        # Analyze confident predictions
        self.analyze_confident_predictions(n_per_class=10)
        
        # Create visualizations for each class
        print("🎨 Creating class-specific visualizations...")
        for class_name in self.label_names:
            # Confident correct
            if len(self.confident_correct[class_name]) > 0:
                self.visualize_class_analysis(class_name, "correct")
            
            # Confident incorrect
            if len(self.confident_incorrect[class_name]) > 0:
                self.visualize_class_analysis(class_name, "incorrect")
        
        # Create summary
        self.create_summary_visualization()
        
        # Print final report
        print("\n" + "="*60)
        print("📊 GRAD-CAM ANALYSIS COMPLETE")
        print("="*60)
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"🏆 Best model F1-score: {self.best_score:.4f}")
        print(f"📈 Test accuracy: {np.mean(self.predictions == self.test_labels):.4f}")
        print("\n📂 Generated files:")
        print(f"  - Summary: summary_analysis.png")
        print(f"  - Class analysis: {len(list((self.output_dir / 'class_analysis').glob('*.png')))} files")
        print("="*60)

def main():
    analyzer = GradCAMBestModelAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 