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
    renaming_map = {}
    for feature_group in harmonization_map:
        canonical_name = feature_group["canonical_name"]
        original_cols = feature_group["original_columns"].get(filename, [])
        for col in original_cols:
            if col in df.columns:
                renaming_map[col] = canonical_name
    
    # Create a new DataFrame with only the canonical columns, in a consistent order
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
                    # Standardize columns before extracting values
                    standardized_df = standardize_dataframe_columns(df, filename, harmonization_map)
                    if canonical_feature_name in standardized_df.columns:
                        unique_values.update(standardized_df[canonical_feature_name].dropna().astype(str).unique())
                    else:
                        print(f"Warning: Canonical column '{canonical_feature_name}' not found in standardized DataFrame for '{filename}'.")
            break

    if not found_feature:
        print(f"Error: Canonical feature '{canonical_feature_name}' not found in the harmonization map.")

    return sorted(list(unique_values))

def merge_datasets_by_canonical_key(harmonization_map, data_folder_path, merge_key_canonical_name):
    """
    Merges all datasets in the specified folder based on a common canonical key.
    Returns a single merged DataFrame.
    """
    merged_df = None
    all_files = [f for f in os.listdir(data_folder_path) if os.path.isfile(os.path.join(data_folder_path, f))]

    for filename in all_files:
        file_path = os.path.join(data_folder_path, filename)
        df = read_dataset(file_path)
        if df is not None:
            standardized_df = standardize_dataframe_columns(df, filename, harmonization_map)
            
            if merge_key_canonical_name not in standardized_df.columns:
                print(f"Warning: Merge key '{merge_key_canonical_name}' not found in standardized DataFrame for '{filename}'. Skipping merge for this file.")
                continue

            if merged_df is None:
                merged_df = standardized_df
            else:
                # Perform an outer merge to keep all data, aligning on the canonical merge key
                merged_df = pd.merge(merged_df, standardized_df, on=merge_key_canonical_name, how='outer', suffixes=('_x', '_y'))
                # Handle duplicate columns introduced by merge (e.g., if two files have 'DrugName' and it's not the merge key)
                # This is a basic handling; more sophisticated logic might be needed for complex merges
                cols_to_drop = [col for col in merged_df.columns if col.endswith(('_x', '_y')) and col[:-2] != merge_key_canonical_name]
                merged_df.drop(columns=cols_to_drop, inplace=True)

    return merged_df

def filter_dataframe_by_canonical_value(df, canonical_feature_name, value):
    """
    Filters a DataFrame based on a specific value in a canonical feature column.
    Assumes the DataFrame already has canonical column names.
    """
    if canonical_feature_name not in df.columns:
        print(f"Error: Canonical feature '{canonical_feature_name}' not found in the DataFrame.")
        return pd.DataFrame() # Return empty DataFrame
    
    # Ensure the value is treated as string for comparison, especially for mixed types
    return df[df[canonical_feature_name].astype(str) == str(value)]

def main():
    parser = argparse.ArgumentParser(description="Perform data manipulation based on a harmonization map.")
    parser.add_argument("harmonization_map_path", type=str,
                        help="Path to the JSON file containing the harmonization map (output from main.py).")
    parser.add_argument("data_folder_path", type=str,
                        help="Path to the folder containing the original datasets.")
    parser.add_argument("--action", type=str, required=True, choices=["unique_values", "merge", "filter"],
                        help="Action to perform: 'unique_values', 'merge', or 'filter'.")
    parser.add_argument("--canonical_feature", type=str,
                        help="The canonical name of the feature for 'unique_values' or 'filter' actions, or the merge key for 'merge' action.")
    parser.add_argument("--filter_value", type=str,
                        help="The value to filter by for the 'filter' action.")

    args = parser.parse_args()

    harmonization_map = load_harmonization_map(args.harmonization_map_path)
    if harmonization_map is None:
        return

    if not os.path.isdir(args.data_folder_path):
        print(f"Error: The data folder path '{args.data_folder_path}' is not a valid directory.")
        return

    if args.action == "unique_values":
        if not args.canonical_feature:
            print("Error: --canonical_feature is required for 'unique_values' action.")
            return
        print(f"Getting unique values for canonical feature: '{args.canonical_feature}'...")
        unique_vals = get_unique_values_for_canonical_feature(
            harmonization_map, args.canonical_feature, args.data_folder_path
        )
        if unique_vals:
            print(f"Unique values for '{args.canonical_feature}':")
            for val in unique_vals:
                print(f"- {val}")
        else:
            print(f"No unique values found for '{args.canonical_feature}'.")

    elif args.action == "merge":
        if not args.canonical_feature:
            print("Error: --canonical_feature (merge key) is required for 'merge' action.")
            return
        print(f"Merging datasets by canonical key: '{args.canonical_feature}'...")
        merged_df = merge_datasets_by_canonical_key(harmonization_map, args.data_folder_path, args.canonical_feature)
        if merged_df is not None and not merged_df.empty:
            print("Merged DataFrame (first 5 rows):")
            print(merged_df.head().to_markdown(index=False))
            print(f"\nMerged DataFrame shape: {merged_df.shape}")
            # You might want to save this merged_df to a file here
            # e.g., merged_df.to_csv("merged_data.csv", index=False)
        else:
            print("No data merged or merged DataFrame is empty.")

    elif args.action == "filter":
        if not args.canonical_feature or not args.filter_value:
            print("Error: --canonical_feature and --filter_value are required for 'filter' action.")
            return
        
        # For filtering, we need to merge all data first to have a single DataFrame to filter
        print(f"Merging all datasets to filter by '{args.canonical_feature}' == '{args.filter_value}'...")
        # A common merge key might be needed here, or merge all and then filter
        # For simplicity, let's assume we merge all data first for filtering example
        # In a real scenario, you might want to load and standardize a single file if filtering only that.
        # For this example, we'll merge all data using the first canonical feature as a dummy merge key if no specific merge key is provided.
        dummy_merge_key = harmonization_map[0]["canonical_name"] if harmonization_map else None
        if not dummy_merge_key:
            print("Error: Cannot perform filter action without a canonical feature to merge on.")
            return

        all_data_df = merge_datasets_by_canonical_key(harmonization_map, args.data_folder_path, dummy_merge_key)
        
        if all_data_df is not None and not all_data_df.empty:
            print(f"Filtering DataFrame by '{args.canonical_feature}' == '{args.filter_value}'...")
            filtered_df = filter_dataframe_by_canonical_value(all_data_df, args.canonical_feature, args.filter_value)
            if not filtered_df.empty:
                print("Filtered DataFrame (first 5 rows):")
                print(filtered_df.head().to_markdown(index=False))
                print(f"\nFiltered DataFrame shape: {filtered_df.shape}")
                # You might want to save this filtered_df to a file here
            else:
                print(f"No data found matching filter: '{args.canonical_feature}' == '{args.filter_value}'.")
        else:
            print("No data available to filter.")

if __name__ == "__main__":
    main()
