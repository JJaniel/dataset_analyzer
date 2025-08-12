import json
import os
import pandas as pd
import polars as pl

def parse_json_with_fix(json_string, retries=3):
    """
    Attempts to parse a JSON string, with retries and basic fixing for common issues.
    """
    for i in range(retries):
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error (attempt {i+1}/{retries}): {e}")
            # Attempt to fix common issues
            if "Unterminated string" in str(e) or "Expecting ',' delimiter" in str(e):
                if json_string.endswith('}'):
                    json_string += ']'
                elif json_string.endswith(']'):
                    json_string += '}'
                else:
                    json_string += '}'
            elif "Expecting property name enclosed in double quotes" in str(e):
                # This is harder to fix automatically without more context
                pass
            
            if i < retries - 1:
                print("Attempting to fix JSON and retry...")
            else:
                raise # Re-raise if all retries fail

    return json.loads(json_string) # Should not be reached if retries fail

def read_dataset_sample(file_path, sample_size=5):
    """
    Reads a dataset file and returns a small sample of it.
    For CSV and Excel files, it returns a pandas DataFrame.
    For other file types, it returns None.
    """
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, nrows=sample_size)
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path, nrows=sample_size)
        else:
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def read_full_dataset(file_path):
    """
    Reads a full dataset file and returns a pandas DataFrame.
    For CSV and Excel files, it returns a pandas DataFrame.
    For other file types, it returns None.
    """
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path)
        else:
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def get_unique_values_sample(df, sample_size=5):
    """
    Gets a sample of unique values from each column of a DataFrame.
    """
    unique_values = {}
    for column in df.columns:
        try:
            unique_vals = df[column].unique()
            # Take a sample of unique values, convert to list for JSON serialization
            unique_values[column] = unique_vals[:sample_size].tolist()
        except Exception as e:
            print(f"Could not get unique values for column {column}: {e}")
            unique_values[column] = []
    return unique_values

def get_file_paths(folder_path, extensions):
    """
    Get all file paths in a folder with given extensions.
    """
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_paths.append(os.path.join(root, file))
    return file_paths
