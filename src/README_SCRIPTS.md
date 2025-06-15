# Duck Classification Scripts

This directory contains improved scripts for duck classification training and analysis with flexible input/output paths.

## Scripts Overview

### 1. F1-Optimized Training (`run_f1_optimized_training.py`)

Trains a duck classification model optimized for F1-macro score using the best hyperparameters.

**Usage:**
```bash
python src/run_f1_optimized_training.py [OPTIONS]
```

**Options:**
- `--data-path`: Path to duck images organized by species (default: `images/Phone`)
- `--output-dir`: Directory for all training outputs (default: `output`)
- `--params-file`: JSON file with best hyperparameters (default: `src/best_model.json`)
- `--verbose`: Enable detailed output

**Examples:**
```bash
# Basic usage with defaults
python src/run_f1_optimized_training.py

# Custom paths
python src/run_f1_optimized_training.py --data-path /path/to/duck/images --output-dir results --verbose

# Different parameter file
python src/run_f1_optimized_training.py --params-file hyperparameter_search/best_params.json
```

**Output Structure:**
```
output/
└── best_model_duck_classification_f1_optimized_YYYYMMDD_HHMMSS/
    ├── model/
    │   ├── best_duck_classifier_f1_optimized.h5
    │   └── model_architecture.json
    ├── plots/
    │   ├── training_history.png
    │   ├── confusion_matrix.png
    │   ├── class_weights.png
    │   └── cross_validation_results.png
    ├── metrics/
    │   └── evaluation_metrics.json
    └── logs/
        └── training.log
```

### 2. GradCAM Analysis (`run_gradcam_analysis.py`)

Analyzes trained models using GradCAM to understand model decision-making.

**Usage:**
```bash
python src/run_gradcam_analysis.py [OPTIONS]
```

**Options:**
- `--data-path`: Path to duck images organized by species (default: `images/Phone`)
- `--output-dir`: Directory for all analysis outputs (default: `output`)
- `--model-path`: Path to trained model file (.h5). Auto-detects if not specified
- `--params-file`: JSON file with model hyperparameters (default: `src/best_model.json`)
- `--verbose`: Enable detailed output

**Examples:**
```bash
# Basic usage with auto-detection
python src/run_gradcam_analysis.py

# Specify model path explicitly
python src/run_gradcam_analysis.py --model-path output/best_model_*/model/best_duck_classifier_f1_optimized.h5

# Custom data and output paths
python src/run_gradcam_analysis.py --data-path /path/to/duck/images --output-dir analysis_results
```

**Output Structure:**
```
output/
└── gradcam_best_model_analysis_YYYYMMDD_HHMMSS/
    ├── confident_correct/
    │   ├── Autre_confident_correct.png
    │   ├── Colvert_femelle_confident_correct.png
    │   └── ...
    ├── confident_incorrect/
    │   ├── Autre_confident_incorrect.png
    │   ├── Colvert_mâle_confident_incorrect.png
    │   └── ...
    └── class_analysis/
        ├── Autre_correct.png
        ├── Autre_incorrect.png
        └── ...
```

## Data Organization

Both scripts expect duck images to be organized by species in subdirectories:

```
images/Phone/
├── Autre/
│   ├── image1.jpg
│   └── image2.jpg
├── Colvert femelle/
│   ├── image1.jpg
│   └── image2.jpg
├── Colvert mâle/
├── Foulque macroule/
└── Grèbe huppé/
```

## Dependencies

Required packages are listed in `requirements.txt`. Install with:
```bash
pip install -r requirements.txt
```

For GradCAM analysis, you also need:
```bash
pip install tf-keras-vis
```

## Workflow

1. **Train Model:**
   ```bash
   python src/run_f1_optimized_training.py --data-path images/Phone --output-dir output --verbose
   ```

2. **Analyze Model:**
   ```bash
   python src/run_gradcam_analysis.py --data-path images/Phone --output-dir output --verbose
   ```

3. **Review Results:** Check the timestamped directories in the output folder for:
   - Training metrics and plots
   - Model files
   - GradCAM visualizations

## Notes

- **Automatic Model Detection:** The GradCAM script automatically finds the most recent model file if not specified
- **Timestamped Outputs:** All outputs are saved in timestamped directories to prevent overwrites
- **Flexible Paths:** Both scripts support custom input/output paths for different project structures
- **Error Handling:** Comprehensive validation and error reporting with helpful suggestions
- **Progress Tracking:** Clear progress indicators and estimated completion times 