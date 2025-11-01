"""Utility helpers for the project."""
import json
import os
import random
import time
from pathlib import Path
import numpy as np


def set_seed(seed):
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class Timer:
    """Simple context manager for timing."""
    
    def __enter__(self):
        self.start = time.time()
        self.end = None
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
    
    @property
    def elapsed(self):
        if self.start is None:
            return 0.0
        reference = self.end if self.end is not None else time.time()
        return reference - self.start


def save_json(data, path):
    """Save dict to JSON."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def save_jsonl(data, path):
    """Save list of dicts to JSONL."""
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def load_json(path):
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def ensure_dir(path):
    """Create the directory for a file or folder path."""
    target = Path(path)
    directory = target if target.suffix == "" else target.parent
    directory.mkdir(parents=True, exist_ok=True)


def get_package_versions():
    """Get package versions for reproducibility."""
    import numpy
    import pandas
    import sklearn
    import xgboost
    
    return {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }
