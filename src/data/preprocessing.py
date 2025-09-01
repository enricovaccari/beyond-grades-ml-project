# -*- coding: utf-8 -*-
# =============================================================================
# Module:        preprocessing.py
# Description:   Common preprocessing utilities and ML pipeline components.
# Version:       1.0.1
# Date:          2025-09-01
# Author:        Enrico Vaccari <e.vaccari99@gmail.com>
# License:       MIT
# Notes:         Intended to be imported from notebooks / scripts.
# =============================================================================

from __future__ import annotations

# Standard library
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# Core scientific stack
import numpy as np
import pandas as pd

# Plotting (optional; keep lightweight usage)
import matplotlib.pyplot as plt

# Scikit-learn: preprocessing & pipelines
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression


# Scikit-learn: models & evaluation
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import KFold, cross_validate, learning_curve, GridSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    make_scorer,
)

# Optional: XGBoost (guarded import)
try:
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:
    XGBRegressor = None  # type: ignore[assignment]
    HAS_XGB = False

# Public API (adjust as you add functions)
__all__ = [
    # preprocessing blocks
    "ColumnTransformer",
    "Pipeline",
    "OneHotEncoder",
    "StandardScaler",
    "SimpleImputer",
    "SelectKBest",
    "f_regression",
    # models
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "SVR",
    "KNeighborsRegressor",
    "XGBRegressor",  # may be None if xgboost not installed
    # utils
    "KFold",
    "cross_validate",
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "make_scorer",
    "np",
    "pd",
    "plt",
    "HAS_XGB",
]

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

def run_model_suite(
    X_train, y_train,
    numeric_features, categorical_features,
    models: dict | None = None,
    k_best: int | None = 12,
    cv_splits: int = 5,
    X_test=None, y_test=None,
):
    """
    Simple end-to-end: CV-compare models -> pick best by RMSE -> refit on full train.
    Optionally evaluate on test. Returns (results_df, best_name, best_pipe, test_metrics).

    - Uses `preprocessing.create_pipeline(numeric_features, categorical_features, k_best, model)`
      so preprocessing + SelectKBest happen inside CV (CV-safe).
    - If X_test and y_test are provided, computes RMSE/MAE/R2 on test for the best model.
    """
    
    # Define models
    if models is None:
        models = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.01, max_iter=10000),
            "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(random_state=42),
            "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
            "KNN": KNeighborsRegressor(n_neighbors=5),
            # XGBoost (safe settings, can tune later)
            "XGBoost": XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                reg_alpha=0.0,
                random_state=42,
                tree_method="hist",    # quick on CPU
                n_jobs=-1
            ),
        }

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # Scorer: in sklearn, names "neg_*" mean higher is better.
    # I will convert the sign later to have RMSE > 0.
    scoring = {
        "neg_rmse": "neg_root_mean_squared_error",
        "mae": make_scorer(mean_absolute_error),
        "r2": make_scorer(r2_score),
    }

    rows = []

    for name, model in models.items():
        pipe = create_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            k_best=k_best,
            model=model
        )

        scores = cross_validate(
            pipe, X_train, y_train.squeeze(),
            cv=kf,
            scoring=scoring,
            return_train_score=False,
            error_score="raise"
        )

        # Invert sign for RMSE to have positive values
        rmse_mean = (-scores["test_neg_rmse"]).mean()
        mae_mean  = scores["test_mae"].mean()
        r2_mean   = scores["test_r2"].mean()
        rows.append((name, rmse_mean, mae_mean, r2_mean))

    results_df = pd.DataFrame(rows, columns=["Model", "RMSE", "MAE", "R2"]).sort_values("RMSE")
    best_name = results_df.iloc[0]["Model"]

    # Refit best on full train
    best_pipe = create_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        k_best=k_best,
        model=models[best_name]
    ).fit(X_train, y_train.squeeze())

    test_metrics = None
    if X_test is not None and y_test is not None:
        y_pred = best_pipe.predict(X_test)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        test_metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}

    return results_df, best_name, best_pipe, test_metrics

# -------------------------------------------------------------------

def plot_learning_curve(pipe, name, X, y, cv, scoring,
                        train_sizes=np.linspace(0.1, 1.0, 6)):
    est = clone(pipe)  # I am not touchig the original pipe that was already fitted elsewhere.
    sizes, train_scores, val_scores = learning_curve(
        estimator=est,
        X=X, y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=1, # stricter determinism
        shuffle=True,
        random_state=42
    )
    tr_mean, tr_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    va_mean, va_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    plt.figure(figsize=(7, 4))
    plt.plot(sizes, tr_mean, label="Train")
    plt.fill_between(sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.2)
    plt.plot(sizes, va_mean, label="Validation")
    plt.fill_between(sizes, va_mean - va_std, va_mean + va_std, alpha=0.2)
    plt.title(f"Learning Curve – {name} ({scoring})")
    plt.xlabel("Number of Training Samples")
    plt.ylabel(scoring)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------

def tune_model(
    X_train, y_train,
    numeric_features, categorical_features,
    model, param_grid,
    k_best=12, cv_splits=5, n_jobs=-1, verbose=0, random_state=42
):
    """
    Grid-search a model inside your create_pipeline (CV-safe).
    Returns: best_pipe (fitted), best_params (dict), results_df (with RMSE), grid (GridSearchCV object).
    """
    # Build CV
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    # Wrap model with your preprocessing + optional SelectKBest
    pipe = create_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        k_best=k_best,
        model=model
    )

    # Version-proof RMSE scorer (negative because GridSearchCV maximizes the score)
    rmse_scorer = make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False)

    # Grid search
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=rmse_scorer,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,            # refit best on full training set automatically
        return_train_score=True,
        verbose=verbose,
        error_score="raise"
    )
    grid.fit(X_train, y_train.squeeze())

    # Results as a tidy DataFrame
    res = pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")
    res["mean_test_RMSE"]  = -res["mean_test_score"]
    res["std_test_RMSE"]   =  res["std_test_score"]
    if "mean_train_score" in res:
        res["mean_train_RMSE"] = -res["mean_train_score"]

    keep_cols = [c for c in [
        "rank_test_score", "mean_test_RMSE", "std_test_RMSE", "mean_train_RMSE", "params"
    ] if c in res.columns]
    results_df = res[keep_cols].reset_index(drop=True)

    best_pipe   = grid.best_estimator_
    best_params = grid.best_params_
    best_rmse   = -grid.best_score_

    print("✅ Best params:", best_params)
    print(f"✅ Best CV RMSE: {best_rmse:.6f}")

    return best_pipe, best_params, results_df, grid

# -------------------------------------------------------------------
