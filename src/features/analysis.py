# -*- coding: utf-8 -*-
#**********************************************************************************
# content       = analysis.py functions
#
# version       = 1.0.0
# date          = 26-08-2025
#
# how to        = gets imported automatically from notebooks
# dependencies  = Python 3, math, typing, pandas, seaborn, matplotlib, scipy
# todos         =
#
# license       = MIT
# author        = Enrico Vaccari <e.vaccari99@gmail.com>
#
# © ALL RIGHTS RESERVED
#**********************************************************************************

# -------------------------------------------------------------------
# LIBRARIES (for analysis.py)
# -------------------------------------------------------------------
import math
from typing import List, Dict, Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import scipy.stats as stats

# -------------------------------------------------------------------
# STYLE
# -------------------------------------------------------------------

# Plot style (coerente in tutto il progetto)
plt.style.use("default")
sns.set_theme(context="notebook", style="whitegrid")


# -------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------

def plot_single_column_distribution(
    series: pd.Series,
    column_type: str,
    bins: int = 30,
    figsize: tuple = (10, 6),
    title: str = None
):
    """
    Plot distribution of a single column, given its type.

    Parameters
    ----------
    series : pd.Series
        The column to plot.
    column_type : str
        Must be either 'numeric' or 'categorical'.
    bins : int
        Number of bins (only for numeric).
    color : str
        Color for bars.
    figsize : tuple
        Size of the figure.
    title : str
        Custom title.
    """

    if column_type not in ['numeric', 'categorical']:
        raise ValueError("column_type must be either 'numeric' or 'categorical'")

    col_name = series.name or "Value"
    series_clean = series.dropna()

    plt.figure(figsize=figsize)

    if column_type == 'numeric':
        # Numeric column
        mean = series_clean.mean()
        median = series_clean.median()

        plt.hist(series_clean, bins=bins, alpha=0.7, color="skyblue")
        plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
        plt.axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}')
        plt.xlabel(col_name)
        plt.ylabel("Frequency")
        plt.title(title or f"Distribution of {col_name}")
        plt.legend()

    elif column_type == 'categorical':
        # Categorical column
        mode_val = series_clean.mode().iloc[0] if not series_clean.mode().empty else None
        ax = sns.countplot(x=series_clean, color="salmon")
        plt.xlabel(col_name)
        plt.ylabel("Count")
        plt.xticks(rotation=30)

        if mode_val is not None:
            # Add vertical line on mode
            try:
                mode_index = list(series_clean.value_counts().index).index(mode_val)
                ax.axvline(x=mode_index, color='purple', linestyle='--', label=f"Mode: {mode_val}")
            except ValueError:
                pass
            plt.title(title or f"{col_name} (Mode: {mode_val})")
        else:
            plt.title(title or f"{col_name} (Mode: N/A)")

        plt.legend()

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------

def plot_box(feature, feature_name):
    """
    Plots a boxplot for a single numerical feature.

    Parameters:
    -----------
    feature : array-like or pd.Series
        The numerical values to plot.
    feature_name : str
        The label to use in the plot.

    Returns:
    --------
    None
    """
    sns.boxplot(y=feature, color="skyblue")
    plt.title(f"Boxplot of {feature_name}")
    plt.ylabel(feature_name)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------

def plot_qq(feature, name):
    """
    Plot a Q-Q (Quantile-Quantile) plot for a numeric pandas Series to assess normality.

    The plot compares the quantiles of the input data to the quantiles of a standard normal distribution.
    If the data is normally distributed, the points will lie roughly on the diagonal line.

    Parameters
    ----------
    feature : pd.Series
        The numeric series to plot.
    name : string
        The name of the feature

    Raises
    ------
    TypeError
        If the input series is not numeric.
    """
    stats.probplot(feature, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot of {name}")
    plt.show()

 # -----------------------------------------------------------------
 #   
def plot_distributions_by_type(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    bins: int = 20,
    max_cols_per_row: int = 4,
    numeric_color: str = "skyblue",
    categorical_color: str = "salmon",
) -> Dict[str, Dict]:
    """
    Plot numeric and categorical feature distributions.
    - Numeric: histogram + mean/median lines + Shapiro test
    - Categorical: count plot + mode line and title
    """

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns and c not in numeric_cols]

    results: Dict[str, Dict] = {"normality": {}}

    # --- NUMERIC ---
    if numeric_cols:
        n = len(numeric_cols)
        n_cols = min(max_cols_per_row, n)
        n_rows = math.ceil(n / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.ravel() if n > 1 else [axes]

        for i, col in enumerate(numeric_cols):
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            ax = axes[i]

            sns.histplot(series, bins=bins, kde=False, ax=ax, color=numeric_color, alpha=0.7)
            mean = series.mean()
            median = series.median()

            ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
            ax.axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}')
            ax.set_xlabel(col)
            ax.set_ylabel("Count")

            # Shapiro test
            if len(series) >= 3:
                sample = series.sample(min(len(series), 5000), random_state=42)
                stat, p = shapiro(sample)
                is_normal = p > 0.05
                skewness = " (skewed)" if abs(mean - median) / series.std() > 0.3 else ""
                label = f"{col}{' (Target)' if col == 'GPA' else ''}\nShapiro p={p:.3f} → {'normal' if is_normal else 'not normal'}{skewness}"
                results["normality"][col] = {"stat": float(stat), "p": float(p), "normal_at_0.05": bool(is_normal)}
            else:
                label = f"{col}{' (Target)' if col == 'GPA' else ''}"

            ax.set_title(label)
            ax.legend()

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    # --- CATEGORICAL ---
    if categorical_cols:
        n = len(categorical_cols)
        n_cols = min(max_cols_per_row, n)
        n_rows = math.ceil(n / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.ravel() if n > 1 else [axes]

        for i, col in enumerate(categorical_cols):
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else None
            ax = axes[i]

            sns.countplot(x=df[col], ax=ax, color=categorical_color)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")

            if mode_val is not None:
                ax.axvline(x=mode_val, color='purple', linestyle='--', label=f"Mode: {mode_val}")
                title = f"{col}\nMode: {mode_val}"
            else:
                title = f"{col}\nMode: N/A"

            ax.set_title(title)
            ax.tick_params(axis="x", rotation=30)
            ax.legend()

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    return results

# -----------------------------------------------------------------

def correlations(dataset: pd.DataFrame, columns_list: List[str]) -> None:
    """
    Plot correlation matrix heatmap of the given numeric columns.

    Parameters
    ----------
    dataset : pd.DataFrame
    columns_list : list of str
        Numeric columns to analyze.
    """
    # controllo: tutte numeriche
    if not all(pd.api.types.is_numeric_dtype(dataset[col]) for col in columns_list):
        print("Not all columns are numeric. Exiting function.")
        return

    corr = dataset[columns_list].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        cbar=True,
    )
    plt.title("Correlation Matrix of Numerical Features")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------

def plot_categorical_target_means(X, y, categorical_cols, max_unique=10, cols_per_row=2):
    """
    Plots barplots of target mean by categorical variable in subplots.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature dataset (train set recommended).
    y : pd.Series
        Target variable (same length as X).
    categorical_cols : list
        List of categorical column names to include.
    max_unique : int
        Max number of unique values a categorical column must have to be included.
    cols_per_row : int
        Number of plots per row in the subplot grid.
    """
    df = X.copy()
    df["target"] = y

    # Filter eligible categorical columns
    valid_cols = [col for col in categorical_cols if df[col].nunique() <= max_unique]
    n_plots = len(valid_cols)
    n_rows = int(np.ceil(n_plots / cols_per_row))

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(cols_per_row * 5, n_rows * 4))
    axes = np.array(axes).reshape(-1)  # Flatten in case of 1 row

    for i, col in enumerate(valid_cols):
        ax = axes[i]
        summary = df.groupby(col)["target"].agg(["count", "mean"]).sort_values("count", ascending=False)

        sns.barplot(x=summary.index, y=summary["mean"], hue=summary.index, ax=ax, palette="Set2", legend=False)
        ax.set_title(f"{col} → Mean Target")
        ax.set_xlabel(col)
        ax.set_ylabel("Mean target")

        # Annotate bar values
        for j, val in enumerate(summary["mean"]):
            ax.text(j, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=8)

    # Hide empty subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------

def plot_core_relationships(df, study_time_col="StudyTimeWeekly", gpa_col="GPA", threshold=1):
    # Scatter + linear fit
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=study_time_col, y=gpa_col, data=df, alpha=0.7)
    sns.regplot(x=study_time_col, y=gpa_col, data=df, scatter=False, color="red", line_kws={"linewidth":2})
    plt.xlabel("Study Time Weekly (hours)")
    plt.ylabel("GPA")
    plt.title("Study Time vs GPA")
    plt.show()

    # Threshold flag (if you still want it)
    if threshold is not None:
        flag_col = "StudyTimeThreshold"
        df = df.copy()
        df[flag_col] = (df[study_time_col] < threshold).astype(int)

        plt.figure(figsize=(8,5))
        sns.boxplot(x=flag_col, y=gpa_col, data=df, palette="Set2")
        plt.xticks([0,1], ["≥ 1 hour", "< 1 hour"])
        plt.xlabel("Study Time Group")
        plt.ylabel("GPA")
        plt.title("GPA by Study Time Group")
        plt.show()

        plt.figure(figsize=(8,5))
        sns.scatterplot(x=study_time_col, y=gpa_col, hue=flag_col, palette={0:"blue", 1:"red"}, alpha=0.7, data=df)
        plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold}h")
        plt.xlabel("Study Time Weekly (hours)")
        plt.ylabel("GPA")
        plt.title("Study Time vs GPA (threshold highlighted)")
        plt.legend()
        plt.show()

# -----------------------------------------------------------------

def boxplot_by_group(
    df, 
    group_col: str, 
    value_col: str, 
    title: str = None, 
    figsize=(8,5), 
    palette="Set2"
):
    """
    Plot a boxplot of `value_col` grouped by `group_col`, 
    with automatic coloring based on number of categories.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    group_col : str
        Column used to split groups (categorical).
    value_col : str
        Numeric column to plot.
    title : str
        Title of the plot (optional).
    figsize : tuple
        Figure size.
    palette : str or list
        Color palette (will adjust to number of groups automatically).
    """
    # categories sorted to keep order consistent
    categories = sorted(df[group_col].dropna().unique())
    n_groups = len(categories)

    plt.figure(figsize=figsize)
    sns.boxplot(x=group_col, y=value_col, data=df, palette=sns.color_palette(palette, n_groups))

    plt.xlabel(group_col)
    plt.ylabel(value_col)
    plt.title(title if title else f"{value_col} by {group_col}")

    plt.show()

# -----------------------------------------------------------------

def iqr_outliers(s):
    """
    Plots a boxplot for a single numerical feature.

    Parameters:
    -----------
    feature : array-like or pd.Series
        The numerical values to plot.
    feature_name : str
        The label to use in the plot.

    Returns:
    --------
    Values outside of the [q1-1.5*iqr, s>q3+1.5*iqr] range
    """
    q1,q3=s.quantile([.25,.75]); iqr=q3-q1
    return s[(s<q1-1.5*iqr)|(s>q3+1.5*iqr)]

# -----------------------------------------------------------------

def scatter_with_threshold(
    df,
    x_col: str,
    y_col: str,
    threshold: float = None,
    threshold_axis: str = "x",
    hue_col: str = None,
    palette={0: "blue", 1: "red"},
    alpha: float = 0.7,
    figsize=(8,5),
    title: str = None
):
    """
    General scatterplot of y_col vs x_col with optional threshold line and hue.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    x_col : str
        Column for x-axis.
    y_col : str
        Column for y-axis.
    threshold : float | None
        Value at which to draw a threshold line (vertical if axis='x', horizontal if axis='y').
    threshold_axis : str
        'x' for vertical line, 'y' for horizontal line.
    hue_col : str | None
        Optional column to color points by (categorical or binary).
    palette : dict or list
        Colors for groups if hue_col is given.
    alpha : float
        Transparency of scatter points.
    figsize : tuple
        Figure size.
    title : str
        Custom plot title (optional).
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=palette if hue_col else None,
        alpha=alpha
    )

    # Add regression trendline (optional)
    sns.regplot(
        data=df,
        x=x_col,
        y=y_col,
        scatter=False,
        color="black",
        line_kws={"linewidth": 2, "linestyle": "--"}
    )

    # Threshold line
    if threshold is not None:
        if threshold_axis == "x":
            plt.axvline(threshold, color="red", linestyle="--", label=f"{x_col} threshold = {threshold}")
        elif threshold_axis == "y":
            plt.axhline(threshold, color="red", linestyle="--", label=f"{y_col} threshold = {threshold}")

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title if title else f"{y_col} vs {x_col}")
    plt.legend()
    plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------

def scatter_with_auto_threshold(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    threshold: float,
    axis: str = "x",                     # "x" -> vertical line; "y" -> horizontal line
    make_groups: bool = True,            # color points by side of the threshold
    left_label: str = "Below threshold", # used when axis="x" (x < thr) or axis="y" (y < thr)
    right_label: str = "At/Above threshold",
    palette=("tab:blue", "tab:red"),
    show_trend: bool = True,
    alpha: float = 0.7,
    figsize=(8, 5),
    title: str | None = None
):
    """
    Scatter plot of y vs x with an automatic threshold on x or y.
    Optionally colors points by which side of the threshold they fall on,
    and names those groups in the legend.

    Parameters
    ----------
    df : pd.DataFrame
    x, y : str
        Column names for axes.
    threshold : float
        Threshold value to draw.
    axis : {"x","y"}
        Where to place the threshold (vertical for "x", horizontal for "y").
    make_groups : bool
        If True, color by side of threshold (no preexisting column needed).
    left_label, right_label : str
        Group names for legend (below vs at/above threshold).
    palette : tuple
        Colors for the two groups.
    show_trend : bool
        If True, overlays a regression trend line.
    alpha : float
        Point transparency.
    figsize : tuple
        Figure size.
    title : str | None
        Custom title; defaults to "y vs x (threshold ...)".
    """
    data = df[[x, y]].copy()

    # Build group labels if requested
    hue = None
    if make_groups:
        if axis == "x":
            mask = data[x] < threshold
        elif axis == "y":
            mask = data[y] < threshold
        else:
            raise ValueError("axis must be 'x' or 'y'.")

        groups = np.where(mask, left_label, right_label)
        data["_threshold_group"] = pd.Categorical(groups, categories=[left_label, right_label])
        hue = "_threshold_group"

    # Plot
    plt.figure(figsize=figsize)
    sns.scatterplot(
        data=data, x=x, y=y,
        hue=hue,
        palette=palette if hue else None,
        alpha=alpha
    )

    # Optional trend line
    if show_trend:
        sns.regplot(data=data, x=x, y=y, scatter=False, color="black",
                    line_kws={"linewidth": 2, "linestyle": "--"})

    # Draw the threshold line
    if axis == "x":
        plt.axvline(threshold, color="red", linestyle="--", label=f"{x} threshold = {threshold}")
    else:  # axis == "y"
        plt.axhline(threshold, color="red", linestyle="--", label=f"{y} threshold = {threshold}")

    # Labels/title
    if title is None:
        title = f"{y} vs {x} (threshold on {axis} = {threshold})"
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)

    # Legend: combine seaborn hue legend with threshold line label
    handles, labels = plt.gca().get_legend_handles_labels()
    if make_groups:
        # ensure both group labels are present in legend
        if axis == "x":
            thr_label = f"{x} threshold = {threshold}"
        else:
            thr_label = f"{y} threshold = {threshold}"
        # add threshold label if not already present
        if thr_label not in labels:
            handles.append(plt.Line2D([0], [0], color="red", linestyle="--"))
            labels.append(thr_label)
        plt.legend(handles=handles, labels=labels, title=None)
    else:
        # if we didn't color by groups, just show the threshold line legend
        plt.legend()

    plt.tight_layout()
    plt.show()
