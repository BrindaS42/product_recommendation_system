import joblib
from pathlib import Path
from ..config import ARTIFACTS_DIR

def save(obj, name: str):
    path = ARTIFACTS_DIR / name
    joblib.dump(obj, path)
    return str(path)

def load(name: str):
    path = ARTIFACTS_DIR / name
    return joblib.load(path)
