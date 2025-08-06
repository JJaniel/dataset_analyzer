import pandas as pd
import json
import os
import argparse

def load_harmonization_map(json_file_path):
    """Loads the harmonization map from a JSON file."""
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
    """Reads a dataset file (CSV or Excel) and returns a pandas DataFrame."""
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

def get_unique_values_for_canonical_feature(harmonization_map, canonical_feature_name, data_folder_path):
    """
    Retrieves all unique values for a given canonical feature across all relevant datasets.
    """
    unique_values = set()
    found_feature = False

    for feature_group in harmonization_map:
        if feature_group.get("canonical_name") == canonical_feature_name:
            found_feature = True
            original_columns_map = feature_group.get("original_columns", {})

            for filename, columns in original_columns_map.items():
                file_path = os.path.join(data_folder_path, filename)
                df = read_dataset(file_path)
                if df is not None:
                    for col in columns:
                        if col in df.columns:
                            unique_values.update(df[col].dropna().astype(str).unique())
                        else:
                            print(f"Warning: Column '{col}' not found in '{filename}'.")
            break

    if not found_feature:
        print(f"Error: Canonical feature '{canonical_feature_name}' not found in the harmonization map.")

    return sorted(list(unique_values))

def main():
    parser = argparse.ArgumentParser(description="Perform data manipulation based on a harmonization map.")
    parser.add_argument("harmonization_map_path", type=str,
                        help="Path to the JSON file containing the harmonization map (output from main.py).")
    parser.add_argument("canonical_feature_name", type=str,
                        help="The canonical name of the feature to extract unique values for.")
    parser.add_argument("data_folder_path", type=str,
                        help="Path to the folder containing the original datasets.")
    
    args = parser.parse_args()

    harmonization_map = load_harmonization_map(args.harmonization_map_path)
    if harmonization_map is None:
        return

    if not os.path.isdir(args.data_folder_path):
        print(f"Error: The data folder path '{args.data_folder_path}' is not a valid directory.")
        return

    print(f"Getting unique values for canonical feature: '{args.canonical_feature_name}'...")
    unique_vals = get_unique_values_for_canonical_feature(
        harmonization_map, args.canonical_feature_name, args.data_folder_path
    )

    if unique_vals:
        print(f"Unique values for '{args.canonical_feature_name}':")
        for val in unique_vals:
            print(f"- {val}")
    else:
        print(f"No unique values found for '{args.canonical_feature_name}'.")

if __name__ == "__main__":
    main()
