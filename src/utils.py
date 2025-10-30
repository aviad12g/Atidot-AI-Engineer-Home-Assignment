"""Utility functions for seeding, timing, IO, and feature name helpers."""
import os
import random
import time
import json
from pathlib import Path
import numpy as np


def set_seed(seed=42):
    """Set all random seeds for reproducibility (including PYTHONHASHSEED)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # XGBoost random_state is set per-model, not globally


class Timer:
    """Context manager and standalone timer."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
    
    @property
    def elapsed(self):
        """Get elapsed time in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, filepath):
    """Save dict to JSON file."""
    ensure_dir(Path(filepath).parent)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def save_jsonl(items, filepath):
    """Save list of dicts to JSONL file."""
    ensure_dir(Path(filepath).parent)
    with open(filepath, 'w') as f:
        for item in items:
            f.write(json.dumps(item) + '\n')


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_jsonl(filepath):
    """Load JSONL file."""
    items = []
    with open(filepath, 'r') as f:
        for line in f:
            items.append(json.loads(line))
    return items


def get_package_versions():
    """Get versions of key packages."""
    import sys
    import numpy
    import scipy
    import pandas
    import sklearn
    import xgboost
    import shap
    import matplotlib
    
    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
        "matplotlib": matplotlib.__version__,
    }


def clean_feature_names(names):
    """Clean feature names for readable SHAP plots."""
    cleaned = []
    for name in names:
        # Remove prefixes from ColumnTransformer
        if '__' in name:
            name = name.split('__', 1)[1]
        # Shorten one-hot encoded names
        name = name.replace('onehotencoder_', '').replace('simpleimputer_', '')
        cleaned.append(name)
    return cleaned

