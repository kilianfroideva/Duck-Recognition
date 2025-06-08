#!/bin/bash

# =============================================================================
# 🚀 Duck Classification Hyperparameter Search - Server Installation & Run
# =============================================================================
# This script installs all dependencies and runs thorough hyperparameter search
# Usage: ./install_and_run.sh
# Expected runtime: 8-12 hours for thorough search
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Banner
echo "
╔══════════════════════════════════════════════════════════════════════════════╗
║                🦆 Duck Classification Hyperparameter Search                  ║
║                        Server Installation & Execution                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"

# Check if we're in the right directory
if [[ ! -f "hyperparameter_optimizer.py" ]]; then
    error "Please run this script from the hyperparameter_search directory"
    error "Current directory: $(pwd)"
    error "Expected files: hyperparameter_optimizer.py, run_search.py, etc."
    exit 1
fi

log "Starting installation and setup..."

# =============================================================================
# 1. SYSTEM CHECKS
# =============================================================================

log "🔍 Performing system checks..."

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    success "Python3 found: $PYTHON_VERSION"
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
    if [[ $PYTHON_VERSION == 3.* ]]; then
        success "Python found: $PYTHON_VERSION"
        PYTHON_CMD="python"
    else
        error "Python 3.7+ required, found: $PYTHON_VERSION"
        exit 1
    fi
else
    error "Python not found. Please install Python 3.7+"
    exit 1
fi

# Check pip
if command -v pip3 &> /dev/null; then
    success "pip3 found"
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    success "pip found"
    PIP_CMD="pip"
else
    error "pip not found. Please install pip"
    exit 1
fi

# Check if data directory exists
DATA_PATH="../images/Phone"
if [[ -d "$DATA_PATH" ]]; then
    success "Data directory found: $DATA_PATH"
    # Count images
    TOTAL_IMAGES=$(find "$DATA_PATH" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)
    success "Total images found: $TOTAL_IMAGES"
    if [[ $TOTAL_IMAGES -lt 50 ]]; then
        warning "Low number of images detected. Consider adding more for better results."
    fi
else
    error "Data directory not found: $DATA_PATH"
    error "Please ensure your duck images are in the correct directory structure:"
    error "  images/Phone/ClassA/*.jpg"
    error "  images/Phone/ClassB/*.jpg"
    error "  etc."
    exit 1
fi

# =============================================================================
# 2. DEPENDENCY INSTALLATION  
# =============================================================================

log "📦 Installing Python dependencies..."

# Create a temporary requirements check
cat > temp_requirements.txt << EOF
scikit-optimize>=0.10.2
tqdm>=4.66.1
matplotlib>=3.8.0
seaborn>=0.13.0
pandas>=2.1.0
numpy>=1.20.3
tensorflow>=2.15.0
scikit-learn>=1.0.0
Pillow>=10.0.0
EOF

# Install packages with error handling
install_package() {
    local package=$1
    log "Installing $package..."
    if $PIP_CMD install "$package" --user --quiet; then
        success "$package installed successfully"
    else
        error "Failed to install $package"
        exit 1
    fi
}

# Install each package individually for better error handling
while IFS= read -r package; do
    if [[ ! -z "$package" && ! "$package" =~ ^# ]]; then
        install_package "$package"
    fi
done < temp_requirements.txt

# Clean up
rm temp_requirements.txt

# =============================================================================
# 3. INSTALLATION VERIFICATION
# =============================================================================

log "🧪 Verifying installation..."

# Test imports
$PYTHON_CMD -c "
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
import skopt
import matplotlib
import seaborn
import PIL
print('✅ All packages imported successfully')
print(f'✅ TensorFlow: {tf.__version__}')
print(f'✅ NumPy: {np.__version__}')
print(f'✅ scikit-optimize: {skopt.__version__}')
" || {
    error "Package verification failed"
    exit 1
}

# Test hyperparameter optimizer import
$PYTHON_CMD -c "
from hyperparameter_optimizer import HyperparameterOptimizer
optimizer = HyperparameterOptimizer.__new__(HyperparameterOptimizer)
search_space = optimizer.define_search_space()
print(f'✅ Hyperparameter optimizer ready: {len(search_space)} parameters')
" || {
    error "Hyperparameter optimizer verification failed"
    exit 1
}

success "All dependencies installed and verified!"

# =============================================================================
# 4. PRE-FLIGHT CHECKS
# =============================================================================

log "🚁 Pre-flight checks..."

# Check available disk space (need at least 2GB for models and results)
AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
if [[ $AVAILABLE_SPACE -lt 2097152 ]]; then  # 2GB in KB
    warning "Low disk space detected. Ensure you have at least 2GB free."
fi

# Check if GPU is available (optional)
if $PYTHON_CMD -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)" 2>/dev/null | grep -q "True"; then
    success "GPU detected - training will be faster"
else
    warning "No GPU detected - training will use CPU (slower but still works)"
fi

# Estimate runtime
log "📊 Search Configuration:"
echo "   • Strategy: thorough_search"
echo "   • Trials: 50 hyperparameter combinations"
echo "   • Max epochs per trial: 20"
echo "   • Expected runtime: 8-12 hours"
echo "   • Output will be saved with timestamp"

# =============================================================================
# 5. FINAL CONFIRMATION
# =============================================================================

echo ""
log "🚀 Ready to start hyperparameter search!"
echo ""
echo "This will:"
echo "  • Test 50 different hyperparameter combinations"
echo "  • Use Bayesian optimization for intelligent search"
echo "  • Save results after each trial (interruption-safe)"
echo "  • Train the best model at the end"
echo "  • Generate comprehensive analysis plots"
echo ""
echo "Expected improvements: 5-15% over default parameters"
echo ""

# Ask for confirmation (skip in non-interactive mode)
if [[ -t 0 ]]; then
    read -p "Continue with thorough hyperparameter search? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Search cancelled by user"
        exit 0
    fi
else
    log "Non-interactive mode detected - proceeding automatically"
fi

# =============================================================================
# 6. RUN HYPERPARAMETER SEARCH
# =============================================================================

log "🔬 Starting thorough hyperparameter search..."

# Create a log file for the search
LOG_FILE="hp_search_$(date +'%Y%m%d_%H%M%S').log"

# Run the search with logging
{
    echo "=== Hyperparameter Search Started: $(date) ==="
    echo "Strategy: thorough_search"
    echo "Expected trials: 50"
    echo "Data path: $DATA_PATH"
    echo "Python: $PYTHON_VERSION"
    echo "====================================="
    echo ""
    
    $PYTHON_CMD run_search.py --strategy thorough_search --data_path "$DATA_PATH"
    
} 2>&1 | tee "$LOG_FILE"

# Check if search completed successfully
if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    🎉 HYPERPARAMETER SEARCH COMPLETED! 🎉                    ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    success "Search completed successfully!"
    success "Log file saved: $LOG_FILE"
    echo ""
    
    # Find the results directory
    RESULTS_DIR=$(ls -d hp_search_* 2>/dev/null | tail -1)
    if [[ -d "$RESULTS_DIR" ]]; then
        success "Results saved in: $RESULTS_DIR"
        echo ""
        log "📊 Key result files:"
        echo "   • $RESULTS_DIR/best_model/best_duck_classifier.h5 - Optimized model"
        echo "   • $RESULTS_DIR/best_model/final_results.json - Performance metrics"
        echo "   • $RESULTS_DIR/logs/search_results.csv - All trial results"
        echo "   • $RESULTS_DIR/plots/search_analysis.png - Analysis plots"
        echo ""
        
        # Show best results if available
        if [[ -f "$RESULTS_DIR/best_model/final_results.json" ]]; then
            log "🏆 Best Results:"
            $PYTHON_CMD -c "
import json
try:
    with open('$RESULTS_DIR/best_model/final_results.json', 'r') as f:
        results = json.load(f)
    print(f\"   • Validation F1: {results.get('validation_f1', 'N/A'):.4f}\")
    print(f\"   • Test F1: {results.get('test_f1', 'N/A'):.4f}\")
    print(f\"   • Test Accuracy: {results.get('test_accuracy', 'N/A'):.4f}\")
    print(f\"   • Total trials: {results.get('total_trials', 'N/A')}\")
except Exception as e:
    print(f\"   Could not parse results: {e}\")
"
        fi
    fi
    
    echo ""
    success "🚀 Your optimized duck classification model is ready!"
    
else
    error "Search failed or was interrupted"
    error "Check the log file for details: $LOG_FILE"
    exit 1
fi

echo ""
log "Installation and search script completed!"
echo "════════════════════════════════════════════════════════════════════════════════" 