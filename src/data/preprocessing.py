# -*- coding: utf-8 -*-
#**********************************************************************************
# content		= preprocessing.py functions
#
# version		= 1.0.0
# date			= 26-08-2025
#
# how to		= gets imported automatically from notebooks
# dependencies	= Python 3
# todos         = 
# 
# license		= MIT
# author		= Enrico Vaccari <e.vaccari99@gmail.com>
#
# © ALL RIGHTS RESERVED
#**********************************************************************************

# -------------------------------------------------------------------
# LIBRARIES (preprocessing.py)
# -------------------------------------------------------------------

import os, sys, importlib, subprocess
from typing import Optional, Tuple, List
from pathlib import Path

import matplotlib.pyplot as plt

# Map import-name -> pip-name where they differ
_PIP_ALIAS = {"sklearn": "scikit-learn"}

required_packages = ["pandas", "numpy", "scipy", "sklearn"]
_loaded = {}

for _pkg in required_packages:
    try:
        _mod = importlib.import_module(_pkg)
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", _PIP_ALIAS.get(_pkg, _pkg)])
            _mod = importlib.import_module(_pkg)
        except subprocess.CalledProcessError:
            _mod = None
    if _mod:
        _loaded[_pkg] = _mod

pd = _loaded.get("pandas")
np = _loaded.get("numpy")
scipy = _loaded.get("scipy")
sklearn = _loaded.get("sklearn")

if pd is None or np is None or sklearn is None:
    raise ImportError("Required packages not available: pandas/numpy/sklearn.")

# sklearn subimports
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------

def create_pipeline(numeric_features, categorical_features, k_best=None, model=None):
    """
    Preprocess:
      - numeric: median impute + StandardScaler
      - categorical: most_frequent impute + OneHotEncoder(ignore unknown)
    Then (optional) SelectKBest(f_regression, k=k_best), then model.
    CV-safe: tutto viene rifittato per ogni fold.
    """
    if model is None:
        model = LinearRegression()

    # Define transformers for preprocessing
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')  # One-hot encode region
    
    # Define imputers 
    numerical_imputer = SimpleImputer(strategy='median')  # Impute with median as numeric features are not normally distributed
    categorical_imputer = SimpleImputer(strategy='most_frequent')  # Impute with most frequent

    # Build preprocessor
    num_pipe = Pipeline([
        ("imputer", numerical_imputer),
        ("scaler", numerical_transformer),
    ])
    cat_pipe = Pipeline([
        ("imputer", categorical_imputer),
        ("ohe", categorical_transformer),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features),
    ])

    selector = SelectKBest(score_func=f_regression, k=k_best) if k_best is not None else "passthrough"

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("select", selector),
        ("model", model),
    ])
    return pipe

# -------------------------------------------------------------------

def get_feature_names_after_preprocess(pipe, numeric_features, categorical_features):
    """
    Column names after preprocessing (original numeric + expanded OHE categorical).
    Call AFTER pipe.fit(...).
    """
    pre = pipe.named_steps["preprocessor"]
    ohe = pre.named_transformers_["cat"].named_steps["ohe"]
    cat_names = ohe.get_feature_names_out(categorical_features)
    return list(numeric_features) + list(cat_names)

# -------------------------------------------------------------------

def get_selected_features(pipe, numeric_features, categorical_features):
    """
    Returns (all_names, scores, mask, selected_names) after fitting.
    If k_best=None, mask is all True and scores are NaN.
    """
    all_names = get_feature_names_after_preprocess(pipe, numeric_features, categorical_features)
    selector = pipe.named_steps.get("select", None)

    if selector in (None, "passthrough"):
        scores = np.full(len(all_names), np.nan)
        mask = np.ones(len(all_names), dtype=bool)
        selected_names = all_names
        return all_names, scores, mask, selected_names

    scores = selector.scores_
    mask = selector.get_support()
    selected_names = [n for n, keep in zip(all_names, mask) if keep]
    return all_names, scores, mask, selected_names

# -------------------------------------------------------------------

def baseline_mean_model(y_train, y_test):
    """
    Baseline: always predict the mean of y_train.
    Returns predictions and metrics (RMSE, MAE, R2).
    """
    # Compute mean of train GPA and create an array of the same shape as y_test, filled with that mean
    baseline_pred = np.full_like(y_test, fill_value=np.mean(y_train), dtype=float)

    # Evaluate metrics
    mse  = mean_squared_error(y_test, baseline_pred)
    rmse = np.sqrt(mse)   # manually take square root
    mae  = mean_absolute_error(y_test, baseline_pred)
    r2   = r2_score(y_test, baseline_pred)

    return baseline_pred, {"RMSE": rmse, "MAE": mae, "R2": r2}

# -------------------------------------------------------------------

def plot_model_r2(results, title="Model Comparison (R² Score)"):
    """
    Plot a bar chart of R² scores for models.
    
    Parameters
    ----------
    results : dict or pd.DataFrame
        - If dict: {model_name: r2_score, ...}
        - If DataFrame: must have columns ["Model", "R2"]
    title : str, optional
        Title of the plot
    """
    # Handle DataFrame input
    if hasattr(results, "to_dict"):  # pandas DataFrame
        data = dict(zip(results["Model"], results["R2"]))
    else:
        data = results  # already a dict
    
    plt.figure(figsize=(9,5))
    plt.bar(data.keys(), data.values(), color="skyblue")
    plt.title(title)
    plt.ylabel("R² Score")
    plt.xticks(rotation=30, ha="right")
    plt.axhline(0, color="red", linestyle="--", linewidth=1, label="Baseline (R²=0)")
    plt.legend()
    plt.show()

# -------------------------------------------------------------------