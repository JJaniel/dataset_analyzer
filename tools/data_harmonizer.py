import pandas as pd
import json
import os
import argparse

def load_harmonization_map(json_file_path):
    """
    Loads the harmonization map from a JSON file.
    """
    try:
        with open(json_file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Harmonization map file not found at {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}. Please ensure it's valid JSON.")
        return None

def read_dataset(file_path):
    """
    Reads a dataset file (CSV or Excel) and returns a pandas DataFrame.
    """
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path)
        else:
            print(f"Unsupported file format for {file_path}. Only .csv, .xlsx, .xls are supported.")
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def standardize_dataframe_columns(df, filename, harmonization_map):
    """
    Renames columns in a DataFrame to canonical names based on the harmonization map.
    Returns a new DataFrame with standardized column names.
    """
    standardized_df = pd.DataFrame()
    for feature_group in harmonization_map:
        canonical_name = feature_group["canonical_name"]
        original_cols = feature_group["original_columns"].get(filename, [])
        
        # Find the first matching original column in the DataFrame
        found_col = None
        for col in original_cols:
            if col in df.columns:
                found_col = col
                break
        
        if found_col:
            standardized_df[canonical_name] = df[found_col]
        else:
            # If no original column is found for this canonical feature in this file, add a column of NaNs
            standardized_df[canonical_name] = pd.NA

    return standardized_df