# 🚀 One-Script Linux Server Installation

## Quick Start
```bash
cd hyperparameter_search
./install_and_run.sh
```

## What It Does
1. ✅ Installs all dependencies
2. ✅ Verifies installation  
3. ✅ Runs thorough search (50 trials)
4. ✅ Saves optimized model

## Runtime: ~8-12 hours

## Background Usage
```bash
nohup ./install_and_run.sh > hp_search.out 2>&1 &
tail -f hp_search.out
```

Expected: 5-15% performance improvement! 