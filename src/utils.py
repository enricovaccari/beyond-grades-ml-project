# LIBRARIES THAT ARE NECESSARY IN utils.py
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------

def get_student_csv_path():
    """
    Return the absolute path of the student CSV file.
    Looks in: 
    1) project_root/data/raw/
    2) current working directory
    3) anywhere under project_root (recursive)
    """

    filename="student_performance_data_2025.csv"
    
    # Try to get project root (one level above this file, or cwd in notebooks)
    try:
        project_root = Path(__file__).resolve().parent.parent
    except NameError:
        project_root = Path.cwd().resolve()

    # 1) Standard location: data/raw
    standard = project_root / "data" / "raw" / filename
    if standard.exists():
        return standard.resolve()

    # 2) Current working directory
    local = Path.cwd() / filename
    if local.exists():
        return local.resolve()

    # 3) Recursive search under project root
    matches = list(project_root.rglob(filename))
    if matches:
        return matches[0].resolve()

    # Not found
    raise FileNotFoundError(
        f"{filename} not found in:\n"
        f" - {standard}\n"
        f" - {local}\n"
        f" - anywhere under {project_root}"
    )

# -----------------------------------------------------------------

def load_student_csv(csv_path):
    """
    Loads the student CSV file into a pandas DataFrame.

    Parameters:
        csv_path (str or Path): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    csv_path = Path(csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df

# -----------------------------------------------------------------

def save_dataset(dataset, filename):
    """
    Saves a DataFrame to the 'data/processed' folder (one level up from current folder).
    Creates the folder if it doesn't exist.
    Supports .xlsx and .csv formats. Overwrites the file if it already exists.

    Parameters:
        dataset (pd.DataFrame): The DataFrame to save.
        filename (str): The filename with extension (e.g., 'data.xlsx' or 'data.csv').
    """
    # Project root = go one level up from current working folder
    project_root = Path.cwd().resolve().parent  

    # Target folder = data/processed
    folder = project_root / "data" / "processed"
    folder.mkdir(parents=True, exist_ok=True)  

    # Full path to file
    path = folder / filename

    # Save based on extension
    if filename.endswith('.xlsx'):
        dataset.to_excel(path, index=False)
    elif filename.endswith('.csv'):
        dataset.to_csv(path, index=False)
    else:
        raise ValueError("Unsupported file format. Use '.xlsx' or '.csv'.")

    print(f"✅ File saved at: {path}")




## 2.2 DEFINE FUNCTIONS

import os
import sys
import shutil
import gdown

def get_student_db_path():
    """
    Ensures the presence of the 'student_data.db' SQLite file in the 'original_dataset' folder.

    Workflow:
    1. Checks if 'original_dataset/student_data.db' exists.
       - If not, checks if 'student_data.db' exists in current dir and moves it into 'original_dataset'.
       - If not, downloads it from Google Drive.
    2. Returns the absolute path to the .db file.
    """
    
    folder_name = "original_dataset"
    file_name = "student_data.db"
    file_id = "1GtZC5GcUr59eccdTobshxZVYYzEt4-de"
    drive_url = f"https://drive.google.com/uc?id={file_id}"

    # Get current working directory
    current_dir = os.getcwd()
    dataset_dir = os.path.join(current_dir, folder_name)
    db_path = os.path.join(dataset_dir, file_name)
    local_db_path = os.path.join(current_dir, file_name)

    # Ensure folder exists
    if not os.path.exists(dataset_dir):
        print(f"Directory '{folder_name}' not found. Creating it...")
        os.makedirs(dataset_dir)

    # Check if file is already in the original_dataset folder
    if not os.path.exists(db_path):
        # Check if it exists in current directory
        if os.path.exists(local_db_path):
            print(f"Found '{file_name}' in current directory. Moving it to '{folder_name}'...")
            shutil.move(local_db_path, db_path)
        else:
            # Download it from Google Drive
            print(f"File '{file_name}' not found. Attempting to download from Google Drive...")
            try:
                gdown.download(drive_url, db_path, quiet=False)
            except Exception as e:
                raise RuntimeError(f"Failed to download the file: {e}")

    # Final check
    if os.path.exists(db_path):
        print("✅ Dataset is ready.")
        return db_path
    else:
        print("❌ Critical Error: Unable to find or download the dataset.")
        sys.exit("Notebook execution aborted due to missing required dataset.")

  
# -----------------------------------------------------------------

def run_sql_query(query, db_path):
    """
    Executes an SQL query on the given SQLite database and returns the result as a DataFrame.

    Parameters:
        query (str): The SQL query to execute.
        db_path (str): Path to the SQLite .db file.

    Returns:
        pd.DataFrame: The result of the query.
    """
    conn = sqlite3.connect(db_path) # works cause this file is saved in the dataset directory
  
    df = pd.read_sql_query("SELECT * FROM student_data", conn)

    conn.close()

    return df

# -----------------------------------------------------------------

def explore_dataset(dataset):
    """
    Explores a dataset printing out relevant info
    Parameters:
        dataset (pd.DataFrame): the dataset you would like to explore
    """
    # Print header 
    print('HEADER-----------------------------------------------------------------')
    print(dataset.head())

    # Print info
    print('\nINFO-----------------------------------------------------------------')
    print(dataset.info())

    # Columns type
    print('\nCOLUMN TYPES-----------------------------------------------------------------')
    print(dataset.dtypes)

    # Columns and Rows Amount
    print('\nROWS AND COLUMNS NUMBER-----------------------------------------------------------------')
    print("Rows:", dataset.shape[0])     # .shape returns a tuple (rows number, column number)
    print("Columns:", dataset.shape[1])

# -----------------------------------------------------------------

def out_columns(dataset):
    """
    Explores columns of a dataset and creates global variables to be used
    (columns, columns_list, numeric_cols, numeric_cols_list, categorical_cols, numeric_cols_list)

    Parameters:
        dataset (pd.DataFrame): the dataset you would creates global variables out of
    """

    print('\nALL COLUMNS NAMES-----------------------------------------------------------------')
    global columns
    columns = dataset.columns
    global columns_list
    columns_list = list(columns)
    print(columns_list)

    print('\nNUMERIC COLUMNS NAMES-----------------------------------------------------------------')
    global numeric_cols
    numeric_cols = dataset.select_dtypes(include='number').columns
    global numeric_cols_list
    numeric_cols_list = list(numeric_cols)
    print(numeric_cols_list)

    print('\nCATEGORICAL COLUMNS NAMES-----------------------------------------------------------------')
    global categorical_cols
    categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns
    global categorical_cols_list
    categorical_cols_list = list(categorical_cols)
    print(categorical_cols_list)

    # Describe columns
    print('\nCOLUMNS DESCRIPTION-----------------------------------------------------------------')
    for col in columns:
        print(col)
        print(dataset[col].describe())
    
    print('\nGLOBAL VARIABLES CREATED-----------------------------------------------------------------')
    print('columns, columns_list, numeric_cols, numeric_cols_list, categorical_cols, numeric_cols_list\n')

# -----------------------------------------------------------------


import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from typing import List, Dict, Optional


import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from typing import List, Dict, Optional


def plot_distributions_by_type(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    check_normality: bool = True,
    bins: int = 20,
    max_cols_per_row: int = 4,
    numeric_color: str = "skyblue",
    categorical_color: str = "salmon"
) -> Dict[str, Dict]:
    """
    Plot distributions for numeric and categorical columns.
    Numeric: histograms (with Shapiro normality test).
    Categorical: bar plots of value counts.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    numeric_cols : list[str] | None
        Columns to treat as numeric. If None, inferred automatically.
    categorical_cols : list[str] | None
        Columns to treat as categorical. If None, inferred automatically.
    check_normality : bool
        Run Shapiro test for numeric cols if True.
    bins : int
        Number of bins for numeric histograms.
    max_cols_per_row : int
        Maximum number of subplots per row.
    numeric_color : str
        Color for numeric plots.
    categorical_color : str
        Color for categorical plots.

    Returns
    -------
    results : dict
        Normality test results for numeric features.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Ensure no overlap
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns and c not in numeric_cols]

    results: Dict[str, Dict] = {"normality": {}}

    # ---- NUMERIC ----
    if numeric_cols:
        n = len(numeric_cols)
        n_cols = min(max_cols_per_row, n)
        n_rows = math.ceil(n / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.ravel() if n > 1 else [axes]

        for i, col in enumerate(numeric_cols):
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            sns.histplot(series, bins=bins, kde=True, ax=axes[i], color=numeric_color)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Count")

            if check_normality and len(series) >= 3:
                stat, p = shapiro(series.sample(min(len(series), 5000), random_state=42))
                results["normality"][col] = {"stat": float(stat), "p": float(p), "normal_at_0.05": bool(p > 0.05)}
                axes[i].set_title(f"{col} (Shapiro p={p:.3f})")
            else:
                axes[i].set_title(f"{col}")

        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.show()

    # ---- CATEGORICAL ----
    if categorical_cols:
        n = len(categorical_cols)
        n_cols = min(max_cols_per_row, n)
        n_rows = math.ceil(n / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.ravel() if n > 1 else [axes]

        for i, col in enumerate(categorical_cols):
            sns.countplot(x=df[col], ax=axes[i], color=categorical_color)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Count")
            axes[i].set_title(f"{col}")
            axes[i].tick_params(axis="x", rotation=30)

        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.show()

    return results



def correlations(dataset, columns_list):
    """
    Plots correlation matrix heatmap of the inputted columns.

    Parameters:
        dataset (pd.DataFrame)
        columns_list (list): list of (numeric columns) columns to analyze
    """

    # check for numeric columns
    if not all(pd.api.types.is_numeric_dtype(dataset[col]) for col in columns_list):
        print("Not all columns are numeric. Exiting function.")
        return
    
    # calculate correlation matrix
    correlation_matrix = dataset[columns_list].corr()

    # --- Plot the heatmap ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, 
                annot=True,          # show the correlation values
                cmap='coolwarm',     # color scheme
                fmt=".2f",           # show two decimal places
                vmin=-1, vmax=1)     # force range from -1 to 1
    
    plt.style.use('default')
    plt.title("Correlation Matrix of Numerical Features")
    plt.tight_layout()
    plt.show()