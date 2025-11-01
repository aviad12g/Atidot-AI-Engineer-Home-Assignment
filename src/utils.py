"""Utility functions."""
import json
import os
import random
import time
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
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start


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
    """Create directory if it doesn't exist."""
    if '/' in path:
        dir_path = path if not path.endswith(('.json', '.csv', '.txt', '.png', '.pkl', '.jsonl')) else path.rsplit('/', 1)[0]
        os.makedirs(dir_path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)


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
