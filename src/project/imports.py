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
import seaborn as sns
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

import json
import math
import joblib
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Literal
from scipy.stats import shapiro, ttest_rel, wilcoxon, pearsonr

# Scikit-learn: preprocessing & pipelines
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# Scikit-learn: models & evaluation
from sklearn.svm import SVR
from sklearn.utils import resample
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


from sklearn.model_selection import (
    KFold, cross_validate, cross_val_score,
    learning_curve, validation_curve, train_test_split, GridSearchCV
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, make_scorer,
)

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
        # print(f"{name}.py functions module imported from: {path if path else 'sys.path'}")
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
    "pd", "np", "sns", "plt", "json", "joblib", "warnings", "datetime",
    # sklearn preprocessing
    "ColumnTransformer", "Pipeline", "SimpleImputer",
    "StandardScaler", "OneHotEncoder", "SelectKBest", "f_regression",
    # sklearn utils
    "clone", "KFold", "GridSearchCV", "cross_validate", "cross_val_score",
    "learning_curve", "validation_curve", "train_test_split", "resample",
    # sklearn metrics
    "mean_squared_error", "mean_absolute_error", "r2_score", "make_scorer", "permutation_importance",
    # sklearn models
    "LinearRegression", "Ridge", "Lasso", "ElasticNet",
    "DecisionTreeRegressor", "RandomForestRegressor", "GradientBoostingRegressor",
    "SVR", "KNeighborsRegressor",
    # typing
    "List", "Dict", "Optional", "Literal",
    # scipystats
    "shapiro", "ttest_rel", "wilcoxon", "pearsonr",
    # math
    "math",
    # local modules (if available)
    "cleaning", "preprocessing", "splitting", "analysis", "utils",
    # paths
    "PROJECT_ROOT", "SRC_PATH", "DATA_PATH", "FEATURES_PATH", "UTILS_PATH",
]

# Add only available local modules dynamically
for name, mod in {
    "cleaning": cleaning,
    "preprocessing": preprocessing,
    "splitting": splitting,
    "analysis": analysis,
    "utils": utils
}.items():
    if mod is not None and name not in __all__:
        __all__.append(name)

print("Imports ready: pd, np, sns, plt, joblib, sklearn, etc.")
print("PROJECT_ROOT:", PROJECT_ROOT)
