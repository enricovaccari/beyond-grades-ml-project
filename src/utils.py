# LIBRARIES THAT ARE NECESSARY IN utils.py
import os
import pandas as pd
from pathlib import Path

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
