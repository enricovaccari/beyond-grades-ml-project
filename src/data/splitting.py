# -*- coding: utf-8 -*-
#**********************************************************************************
# content		= splitting.py functions
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
# LIBRARIES (for splitting.py)
# -------------------------------------------------------------------

import sys, importlib, subprocess
from typing import Optional, Tuple
from pathlib import Path

required_packages = ["pandas", "numpy", "sklearn"]  # use sklearn, not scikit-learn
loaded = {}

for pkg in required_packages:
    try:
        mod = importlib.import_module(pkg)
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            mod = importlib.import_module(pkg)
        except subprocess.CalledProcessError:
            mod = None
    if mod:
        loaded[pkg] = mod

# Aliases
pd = loaded.get("pandas")
np = loaded.get("numpy")
sklearn = loaded.get("sklearn")

if sklearn is not None:
    from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------

def quantile_bin_edges(y: pd.Series, q: int) -> Optional[np.ndarray]:
    """
    Return bin edges corresponding to q-quantiles of y, suitable for pd.cut.
    Adds -inf/inf guards so every value falls into a bin.
    """
    try:
        qs = np.linspace(0, 1, q + 1)
        edges = np.quantile(y.astype(float), qs)
        # remove duplicates (can happen with ties); keep order
        edges = np.unique(edges)
        if len(edges) < 3:  # need at least 2 intervals
            return None
        # extend to cover all values robustly
        eps = 1e-9
        edges[0] = edges[0] - eps
        edges[-1] = edges[-1] + eps
        return edges
    except Exception:
        return None

# -------------------------------------------------------------------

def make_strat_bins(y, max_q: int = 5) -> Tuple[Optional[pd.Series], Optional[int], Optional[np.ndarray]]:
    """
    Try q-quantile binning (from max_q down to 2) so each bin has >=2 samples.
    Returns:
        y_binned (Series or None), used_q (int or None), bin_edges (np.ndarray or None)
    """
    y_series = y if isinstance(y, pd.Series) else pd.Series(y, name="target")

    for q in range(max_q, 1, -1):
        try:
            # derive consistent edges once, then cut with them
            edges = quantile_bin_edges(y_series, q)
            if edges is None:
                continue
            # labels 0..(num_bins-1)
            yb = pd.cut(y_series, bins=edges, labels=False, include_lowest=True)
            counts = yb.value_counts(dropna=False)
            if (counts >= 2).all():
                return yb, q, edges
        except Exception:
            continue
    return None, None, None  # not possible

# -------------------------------------------------------------------

def for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to object for Excel writers."""
    return df.apply(lambda s: s.astype(object) if str(s.dtype) == "category" else s)

# -------------------------------------------------------------------

def safe_train_test_split(
    X, y, test_size: float = 0.2, random_state: int = 42, max_q: int = 5, verbose: bool = True
):
    """
    Use quantile stratification if feasible; otherwise fall back to no stratify.
    Returns split plus metadata dict with used_q and bin_edges (if any).
    """
    y_binned, used_q, bin_edges = make_strat_bins(y, max_q=max_q)
    if y_binned is None:
        if verbose:
            print("Stratification not possible → continuing without stratify")
        stratify_arg = None
    else:
        if verbose:
            print(f"Stratification by quantiles: q={used_q}")
        stratify_arg = y_binned

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )
    return X_train, X_test, y_train, y_test, {"used_q": used_q, "bin_edges": bin_edges}

# -------------------------------------------------------------------