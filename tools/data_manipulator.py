import polars as pl
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
    Reads a dataset file (CSV or Excel) and returns a Polars DataFrame.
    """
    try:
        if file_path.endswith('.csv'):
            return pl.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            # Polars requires 'xlsx2csv' or similar for direct Excel reading
            # For simplicity, we'll use pandas to read and then convert to Polars
            # For production, consider direct Polars Excel readers if available or convert to CSV first
            try:
                import pandas as pd
                return pl.from_pandas(pd.read_excel(file_path))
            except ImportError:
                print("Error: pandas is required for reading Excel files. Please install it (`pip install pandas`).")
                return None
        else:
            print(f"Unsupported file format for {file_path}. Only .csv, .xlsx, .xls are supported.")
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def standardize_dataframe_columns(df: pl.DataFrame, filename: str, harmonization_map: list):
    """
    Renames columns in a Polars DataFrame to canonical names based on the harmonization map.
    Returns a new DataFrame with standardized column names and only canonical features.
    """
    # Create a list of expressions for selecting and renaming columns
    select_exprs = []
    for feature_group in harmonization_map:
        canonical_name = feature_group["canonical_name"]
        original_cols = feature_group["original_columns"].get(filename, [])
        
        found_col_expr = None
        for col in original_cols:
            if col in df.columns:
                found_col_expr = pl.col(col).alias(canonical_name)
                break
        
        if found_col_expr is not None:
            select_exprs.append(found_col_expr)
        else:
            # If no original column is found, add a null column with the canonical name
            select_exprs.append(pl.lit(None).alias(canonical_name))

    return df.select(select_exprs)

def get_unique_values_for_canonical_feature(harmonization_map: list, canonical_feature_name: str, data_folder_path: str):
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
                    standardized_df = standardize_dataframe_columns(df, filename, harmonization_map)
                    if canonical_feature_name in standardized_df.columns:
                        # Use .unique() on the Series and convert to string for consistency
                        unique_values.update(standardized_df[canonical_feature_name].cast(pl.Utf8).unique().to_list())
                    else:
                        print(f"Warning: Canonical column '{canonical_feature_name}' not found in standardized DataFrame for '{filename}'.")
            break

    if not found_feature:
        print(f"Error: Canonical feature '{canonical_feature_name}' not found in the harmonization map.")

    return sorted(list(unique_values))

def merge_datasets_by_canonical_key(harmonization_map: list, data_folder_path: str, merge_key_canonical_name: str):
    """
    Merges all datasets in the specified folder based on a common canonical key.
    Returns a single merged Polars DataFrame.
    """
    merged_pl_df = None
    all_files = [f for f in os.listdir(data_folder_path) if os.path.isfile(os.path.join(data_folder_path, f))]

    for filename in all_files:
        file_path = os.path.join(data_folder_path, filename)
        pl_df = read_dataset(file_path)
        if pl_df is not None:
            standardized_pl_df = standardize_dataframe_columns(pl_df, filename, harmonization_map)
            
            if merge_key_canonical_name not in standardized_pl_df.columns:
                print(f"Warning: Merge key '{merge_key_canonical_name}' not found in standardized DataFrame for '{filename}'. Skipping merge for this file.")
                continue

            if merged_pl_df is None:
                merged_pl_df = standardized_pl_df
            else:
                # Polars join, handling potential duplicate columns by selecting distinct ones
                # This is a simplified approach; more complex scenarios might need careful column selection
                merged_pl_df = merged_pl_df.join(standardized_pl_df, on=merge_key_canonical_name, how='outer', suffix=f"_from_{filename.replace('.', '_')}")
                
    return merged_pl_df

def filter_dataframe_by_canonical_value(df: pl.DataFrame, canonical_feature_name: str, value: str):
    """
    Filters a Polars DataFrame based on a specific value in a canonical feature column.
    Assumes the DataFrame already has canonical column names.
    """
    if canonical_feature_name not in df.columns:
        print(f"Error: Canonical feature '{canonical_feature_name}' not found in the DataFrame.")
        return pl.DataFrame() # Return empty DataFrame
    
    # Cast to Utf8 for consistent string comparison
    return df.filter(pl.col(canonical_feature_name).cast(pl.Utf8) == value)

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
        if merged_df is not None and not merged_df.is_empty():
            print("Merged DataFrame (first 5 rows):")
            print(merged_df.head().to_pandas().to_markdown(index=False)) # Convert to pandas for markdown printing
            print(f"\nMerged DataFrame shape: {merged_df.shape}")
            # You might want to save this merged_df to a file here
            # e.g., merged_df.write_csv("merged_data.csv")
        else:
            print("No data merged or merged DataFrame is empty.")

    elif args.action == "filter":
        if not args.canonical_feature or not args.filter_value:
            print("Error: --canonical_feature and --filter_value are required for 'filter' action.")
            return
        
        # For filtering, we need to merge all data first to have a single DataFrame to filter
        print(f"Merging all datasets to filter by '{args.canonical_feature}' == '{args.filter_value}'...")
        # A common merge key might be needed here, or merge all and then filter
        # For this example, we'll merge all data using the first canonical feature as a dummy merge key if no specific merge key is provided.
        dummy_merge_key = harmonization_map[0]["canonical_name"] if harmonization_map else None
        if not dummy_merge_key:
            print("Error: Cannot perform filter action without a canonical feature to merge on.")
            return

        all_data_df = merge_datasets_by_canonical_key(harmonization_map, args.data_folder_path, dummy_merge_key)
        
        if all_data_df is not None and not all_data_df.is_empty():
            print(f"Filtering DataFrame by '{args.canonical_feature}' == '{args.filter_value}'...")
            filtered_df = filter_dataframe_by_canonical_value(all_data_df, args.canonical_feature, args.filter_value)
            if not filtered_df.is_empty():
                print("Filtered DataFrame (first 5 rows):")
                print(filtered_df.head().to_pandas().to_markdown(index=False)) # Convert to pandas for markdown printing
                print(f"\nFiltered DataFrame shape: {filtered_df.shape}")
                # You might want to save this filtered_df to a file here
            else:
                print(f"No data found matching filter: '{args.canonical_feature}' == '{args.filter_value}'.")
        else:
            print("No data available to filter.")

if __name__ == "__main__":
    main()