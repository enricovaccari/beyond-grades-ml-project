# -*- coding: utf-8 -*-
#**********************************************************************************
# content		= utils.py functions
#
# version		= 1.0.0
# date			= 26-08-2025
#
# how to		= gets imported automatically from notebooks
# dependencies	= Python 3, os, sys, shutil, pandas, pathlib
# todos         = 
# 
# license		= MIT
# author		= Enrico Vaccari <e.vaccari99@gmail.com>
#
# © ALL RIGHTS RESERVED
#**********************************************************************************

# -------------------------------------------------------------------
# LIBRARIES (for utils.py)
# -------------------------------------------------------------------

# --- Standard library ---
import os
import sys
import shutil
import importlib
import subprocess
from pathlib import Path
from typing import Dict, Any

# --- Third-party: core scientific ---
import numpy as np

# --- Third-party: scikit-learn ---
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression

# --- Dynamic package import/install ---
required_packages = ["pandas", "gdown"]
loaded = {}

for package in required_packages:
    try:
        mod = importlib.import_module(package)
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            mod = importlib.import_module(package)
        except subprocess.CalledProcessError:
            mod = None
    if mod:
        loaded[package] = mod

# Aliases for convenience
pd = loaded.get("pandas")
gdown = loaded.get("gdown")

# -------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------

def get_student_csv_path() -> Path:
    """
    Return the absolute path of the student CSV file.
    Starting from this file (utils.py), goes up to project root,
    then enters data/raw/ to find the dataset.
    """

    filename = "student_performance_data_2025.csv"

    try:
        # utils.py → utilities → src → project_root
        project_root = Path(__file__).resolve().parent.parent.parent
    except NameError:
        # In notebooks, fallback to cwd
        project_root = Path.cwd().resolve()

    # Path: project_root/data/raw/filename
    dataset_path = project_root / "data" / "raw" / filename
    if dataset_path.exists():
        return dataset_path.resolve()

    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

# -----------------------------------------------------------------

def load_student_dataset(file_path):
    """
    Loads a student dataset from CSV or Excel into a pandas DataFrame.

    Parameters
    ----------
    file_path : str | Path
        Path to the CSV (.csv) or Excel (.xlsx/.xls) file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use '.csv', '.xlsx', or '.xls'.")

    return df

# -----------------------------------------------------------------

def save_dataset(dataset, relative_path):
    """
    Save a dataset under <PROJECT_ROOT>/data/<relative_path>.

    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame (or Series) to save
    relative_path : str
        Subfolder and filename relative to <PROJECT_ROOT>/data,
        e.g. "processed/04_X_train.xlsx" or "outputs/results.csv"

    Notes
    -----
    - Always writes inside the 'data' folder at project root.
    - Creates subfolders if they don't exist.
    - Supports .xlsx and .csv formats.
    """
    # Project root = one level up from current working folder (notebook is in /notebooks)
    project_root = Path.cwd().resolve().parent
    data_root = project_root / "data"

    # Full path
    path = data_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)  # ensure subfolders exist

    # Save
    if path.suffix == ".xlsx":
        dataset.to_excel(path, index=False)
    elif path.suffix == ".csv":
        dataset.to_csv(path, index=False)
    else:
        raise ValueError("Unsupported file format. Use '.xlsx' or '.csv'.")

    print(f"✅ File saved at: {path}")

# -----------------------------------------------------------------

def create_separator(file_path):
    """
    Create a separator .txt file at the given path.

    Parameters
    ----------
    file_path : str | Path
        Full path including the file name, must end with .txt

    Example
    -------
    create_separator("../data/processed/_____DATA_SPLIT_DONE_____.txt")
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists

    file_path.write_text(f"=== {file_path.stem} ===\n")
    print(f"📁 Separator created: {file_path}")

# -----------------------------------------------------------------

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

def save_report_md(report: Dict[str, Any], filename: str) -> Path:
    """
    Save a dictionary report as a Markdown file in ../outputs/reports.

    Parameters
    ----------
    report : dict
        The report dictionary to save.
    filename : str
        The file name with extension (must end with '.md').

    Returns
    -------
    Path
        Absolute path to the saved Markdown file.
    """
    if not filename.lower().endswith(".md"):
        raise ValueError("Filename must end with '.md'")

    save_path = Path("..") / "outputs" / "reports"
    save_path.mkdir(parents=True, exist_ok=True)

    report_file = save_path / filename

    def _dict_to_md(d: Dict[str, Any], level: int = 1) -> str:
        lines = []
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{'#'*(level+1)} {k}")
                lines.append(_dict_to_md(v, level+1))
            else:
                lines.append(f"- **{k}**: {v}")
        return "\n".join(lines)

    md_content = f"# Data Quality Report\n\n{_dict_to_md(report)}\n"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Report saved to: {report_file.resolve()}")
    return report_file.resolve()