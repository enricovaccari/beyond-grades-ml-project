# src/project/project_imports.py
"""
Centralized imports and project setup for Jupyter notebooks.
Works with structure:
project-root/
  ├─ src/
  │   ├─ data/          (cleaning.py, preprocessing.py, splitting.py, ...)
  │   ├─ features/      (analysis.py, ...)
  │   ├─ project/       (project_imports.py)
  │   └─ utilities/     (utils.py, ...)
  └─ notebooks/
"""

import sys
import importlib
from pathlib import Path

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]   # from src/project/ → root
except NameError:  # running inside Jupyter
    PROJECT_ROOT = Path.cwd().resolve().parent

SRC_PATH       = PROJECT_ROOT / "src"
DATA_PATH      = SRC_PATH / "data"
FEATURES_PATH  = SRC_PATH / "features"
UTILS_PATH     = SRC_PATH / "utilities"

# Ensure all relevant folders are in sys.path
for p in (SRC_PATH, DATA_PATH, FEATURES_PATH, UTILS_PATH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ---------------------------------------------------------
# Core scientific stack
# ---------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression

# Plot style
plt.style.use("default")
sns.set_theme(context="notebook", style="whitegrid")

# ---------------------------------------------------------
# Try to import local modules
# ---------------------------------------------------------
def _try_import(name, path=None, reload=True):
    try:
        mod = importlib.import_module(name)
        if reload:
            importlib.reload(mod)
        print(f"{name}.py functions module imported from: {path if path else 'sys.path'}")
        return mod
    except Exception as e:
        print(f"{name} not found or failed to import ({e})")
        return None

cleaning       = _try_import("cleaning",      DATA_PATH)
preprocessing  = _try_import("preprocessing", DATA_PATH)
splitting      = _try_import("splitting",     DATA_PATH)
analysis       = _try_import("analysis",      FEATURES_PATH)
utils          = _try_import("utils",         UTILS_PATH)

# ---------------------------------------------------------
# Final export list
# ---------------------------------------------------------
__all__ = [
    # libs
    "pd", "np", "sns", "plt",
    "ColumnTransformer", "Pipeline", "SimpleImputer",
    "train_test_split", "StandardScaler", "OneHotEncoder",
    "SelectKBest", "f_regression",
    # paths
    "PROJECT_ROOT", "SRC_PATH", "DATA_PATH", "FEATURES_PATH", "UTILS_PATH",
]

# add only available local modules
for name, mod in {
    "cleaning": cleaning,
    "preprocessing": preprocessing,
    "splitting": splitting,
    "analysis": analysis,
    "utils": utils
}.items():
    if mod is not None:
        __all__.append(name)

print("Imports ready: pd, np, sns, plt, train_test_split, etc.")
print("PROJECT_ROOT:", PROJECT_ROOT)
