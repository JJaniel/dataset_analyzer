import polars as pl
import json
import os
import argparse
from tools.llm_manager import get_llm_response
from tools.data_analyzer import analyze_individual_dataset
from tools.data_synthesizer import synthesize_analyses
from langchain.prompts import PromptTemplate
from tools.utils import read_dataset_sample

import inspect

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
            return pl.read_csv(file_path, ignore_errors=True)
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

def standardize_dataframe_columns(df: pl.DataFrame, filename: str, harmonization_map: list, verbose: bool = False):
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

def get_unique_values_for_canonical_feature(harmonization_map: list, canonical_feature_name: str, data_folder_path: str, verbose: bool = False):
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
                    standardized_df = standardize_dataframe_columns(df, filename, harmonization_map, verbose=verbose)
                    if canonical_feature_name in standardized_df.columns:
                        # Use .unique() on the Series and convert to string for consistency
                        unique_values.update(standardized_df[canonical_feature_name].cast(pl.Utf8).unique().to_list())
                    else:
                        if verbose:
                            print(f"Warning: Canonical column '{canonical_feature_name}' not found in standardized DataFrame for '{filename}'.")
            break

    if not found_feature:
        if verbose:
            print(f"Error: Canonical feature '{canonical_feature_name}' not found in the harmonization map.")

    return sorted(list(unique_values))

def merge_datasets_by_canonical_key(harmonization_map: list, data_folder_path: str, merge_key_canonical_name: str, verbose: bool = False):
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
            standardized_pl_df = standardize_dataframe_columns(pl_df, filename, harmonization_map, verbose=verbose)
            
            if merge_key_canonical_name not in standardized_pl_df.columns:
                if verbose:
                    print(f"Warning: Merge key '{merge_key_canonical_name}' not found in standardized DataFrame for '{filename}'. Skipping merge for this file.")
                continue

            if merged_pl_df is None:
                merged_pl_df = standardized_pl_df
            else:
                # Polars join, handling potential duplicate columns by selecting distinct ones
                # This is a simplified approach; more complex scenarios might need careful column selection
                merged_pl_df = merged_pl_df.join(standardized_pl_df, on=merge_key_canonical_name, how='outer', suffix=f"_from_{filename.replace('.', '_')}")
                
    return merged_pl_df

def filter_dataframe_by_canonical_value(df: pl.DataFrame, canonical_feature_name: str, value: str, verbose: bool = False):
    """
    Filters a Polars DataFrame based on a specific value in a canonical feature column.
    Assumes the DataFrame already has canonical column names.
    """
    if canonical_feature_name not in df.columns:
        if verbose:
            print(f"Error: Canonical feature '{canonical_feature_name}' not found in the DataFrame.")
        return pl.DataFrame() # Return empty DataFrame
    
    # Cast to Utf8 for consistent string comparison
    return df.filter(pl.col(canonical_feature_name).cast(pl.Utf8) == value)

def get_unique_values_from_df(df: pl.DataFrame, canonical_feature_name: str, verbose: bool = False):
    """
    Retrieves all unique values for a given canonical feature from a Polars DataFrame.
    """
    if canonical_feature_name not in df.columns:
        if verbose:
            print(f"Error: Canonical feature '{canonical_feature_name}' not found in the DataFrame.")
        return []
    return df[canonical_feature_name].unique().to_list()

def get_unique_values_from_single_file(file_path: str, harmonization_map: list, canonical_feature_name: str, verbose: bool = False):
    """
    Retrieves all unique values for a given canonical feature from a single specified dataset file.
    """
    df = read_dataset(file_path)
    if df is None:
        if verbose:
            print(f"Error: Could not read dataset from {file_path}.")
        return []
    
    # Extract filename from file_path for standardize_dataframe_columns
    filename = os.path.basename(file_path)
    standardized_df = standardize_dataframe_columns(df, filename, harmonization_map, verbose=verbose)
    
    if canonical_feature_name not in standardized_df.columns:
        if verbose:
            print(f"Error: Canonical feature '{canonical_feature_name}' not found in the standardized DataFrame for {filename}.")
        return []
    
    return standardized_df[canonical_feature_name].cast(pl.Utf8).unique().to_list()

def plan_and_execute_manipulation(harmonization_map: list, data_folder_path: str, user_request: str, llm_providers: list, cli_args):
    """
    Uses an LLM (Planner Agent) to generate a data manipulation plan and then executes it.
    """
    available_functions_description = """
    You have the following Polars-based data manipulation functions available:

    1.  `standardize_dataframe_columns(df, filename, harmonization_map)`:
        -   **Description**: Renames columns in a DataFrame to canonical names based on the harmonization map. Returns a new DataFrame with standardized column names and only canonical features.
        -   **Inputs**: `df` (Polars DataFrame), `filename` (string, original filename of the DataFrame), `harmonization_map` (list, the harmonization map).
        -   **Output**: A new Polars DataFrame with standardized columns.

    2.  `get_unique_values_for_canonical_feature(harmonization_map, canonical_feature_name, data_folder_path)`:
        -   **Description**: Retrieves all unique values for a given canonical feature across all relevant datasets.
        -   **Inputs**: `harmonization_map` (list), `canonical_feature_name` (string), `data_folder_path` (string).
        -   **Output**: A sorted list of unique values (strings).

    3.  `merge_datasets_by_canonical_key(harmonization_map, data_folder_path, merge_key_canonical_name)`:
        -   **Description**: Merges all datasets in the specified folder based on a common canonical key. Returns a single merged Polars DataFrame.
        -   **Inputs**: `harmonization_map` (list), `data_folder_path` (string), `merge_key_canonical_name` (string).
        -   **Output**: A merged Polars DataFrame.

    4.  `filter_dataframe_by_canonical_value(df, canonical_feature_name, value)`:
        -   **Description**: Filters a Polars DataFrame based on a specific value in a canonical feature column. Assumes the DataFrame already has canonical column names.
        -   **Inputs**: `df` (Polars DataFrame), `canonical_feature_name` (string), `value` (string).
        -   **Output**: A filtered Polars DataFrame.

    5.  `get_unique_values_from_df(df, canonical_feature_name)`:
        -   **Description**: Retrieves all unique values for a given canonical feature from a Polars DataFrame.
        -   **Inputs**: `df` (Polars DataFrame), `canonical_feature_name` (string).
        -   **Output**: A list of unique values (strings).

    Your task is to generate a JSON plan that describes the sequence of operations to fulfill the user's request. The plan should be a list of objects, where each object represents a step. Each step must have:
    -   `"function"`: The name of the function to call (e.g., "merge_datasets_by_canonical_key").
    -   `"args"`: An object containing the arguments for the function. Arguments should be directly mappable to the function's parameters. For `df` arguments, use a placeholder like `"_current_df_"` if the DataFrame is an output from a previous step, or `"_all_datasets_"` if it implies loading all datasets.
    -   `"output_variable"`: (Optional) The name of a variable to store the output of this step (e.g., "merged_data").

    The harmonization map for the current datasets is provided below. You MUST use the `canonical_name` values from this map when referring to features in your plan. Do NOT use original column names or make up new canonical names.

Harmonization Map: {harmonization_map_json}

When the user asks to find unique values for a canonical feature across all datasets, you should directly use the `get_unique_values_for_canonical_feature` function. Do NOT attempt to merge datasets first for this specific type of request, as it can be inefficient.

Example Plan (Get Unique Values for a Canonical Feature):
    ```json
    [
      {{
        "function": "get_unique_values_for_canonical_feature",
        "args": {{
          "harmonization_map": "_harmonization_map_",
          "canonical_feature_name": "<CANONICAL_FEATURE_NAME_FROM_MAP>",
          "data_folder_path": "_data_folder_path_"
        }},
        "output_variable": "unique_values_for_feature"
      }}
    ]
    ```

Example Plan (Merge and then Filter):
    ```json
    [
      {{
        "function": "merge_datasets_by_canonical_key",
        "args": {{
          "harmonization_map": "_harmonization_map_",
          "data_folder_path": "_data_folder_path_",
          "merge_key_canonical_name": "<CANONICAL_MERGE_KEY_FROM_MAP>"
        }},
        "output_variable": "merged_df"
      }},
      {{
        "function": "filter_dataframe_by_canonical_value",
        "args": {{
          "df": "_merged_df_",
          "canonical_feature_name": "<CANONICAL_FEATURE_NAME_FROM_MAP>",
          "value": "<VALUE_TO_FILTER_BY>"
        }},
        "output_variable": "filtered_df"
      }}
    ]
    ```

User Request: {user_request}
    """

    planner_prompt = PromptTemplate(template=available_functions_description, input_variables=["user_request", "harmonization_map_json"])
    
    print("Generating manipulation plan with LLM...")
    harmonization_map_json = json.dumps(harmonization_map, indent=2)
    if cli_args.verbose:
        print(f"Prompting LLM with: {planner_prompt.template.format(user_request=user_request, harmonization_map_json=harmonization_map_json)}")
    try:
        plan_json_str = get_llm_response(planner_prompt.template, {"user_request": user_request, "harmonization_map_json": harmonization_map_json}, llm_providers)
        if cli_args.verbose:
            print(f"Raw LLM response: {plan_json_str}") # Debug print
        plan_json_str = plan_json_str.strip()
        # Extract only the last JSON part from the LLM response
        json_start = plan_json_str.rfind('[')
        json_end = plan_json_str.rfind(']')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            plan_json_str = plan_json_str[json_start : json_end + 1]
        else:
            raise ValueError("Could not find a valid JSON array in the LLM response.")
        plan = json.loads(plan_json_str)
        print("Generated Plan:")
        print(json.dumps(plan, indent=2))
    except Exception as e:
        print(f"Error generating or parsing LLM plan: {e}")
        return

    # --- Plan Execution ---
    print("Executing manipulation plan...")
    local_vars = {
        "_harmonization_map_": harmonization_map,
        "_data_folder_path_": data_folder_path,
        "_current_df_": None # Placeholder for the current DataFrame being processed
    }

    for step_idx, step in enumerate(plan):
        function_name = step.get("function")
        args = step.get("args", {})
        output_variable = step.get("output_variable")

        if function_name not in globals() and function_name not in locals():
            print(f"Error: Function '{function_name}' not found for step {step_idx + 1}.")
            return

        # Resolve arguments
        resolved_args = {}
        for arg_name, arg_value in args.items():
            if isinstance(arg_value, str) and arg_value.startswith("_") and arg_value.endswith("_"):
                # Resolve placeholders
                if arg_value == "_harmonization_map_":
                    resolved_args[arg_name] = harmonization_map
                elif arg_value == "_data_folder_path_":
                    resolved_args[arg_name] = data_folder_path
                elif arg_value == "_current_df_":
                    resolved_args[arg_name] = local_vars["_current_df_"]
                elif arg_value.startswith("_") and arg_value.endswith("_") and arg_value[1:-1] in local_vars:
                    resolved_args[arg_name] = local_vars[arg_value[1:-1]]
                else:
                    print(f"Warning: Unresolved placeholder '{arg_value}' for argument '{arg_name}' in step {step_idx + 1}. Using as literal.")
                    resolved_args[arg_name] = arg_value
            else:
                resolved_args[arg_name] = arg_value

        try:
            # Get the function object
            func = globals().get(function_name) or locals().get(function_name)
            if func is None:
                raise ValueError(f"Function '{function_name}' not found.")

            if args.verbose:
                print(f"Executing step {step_idx + 1}: {function_name}({resolved_args})")
            
            # Dynamically add verbose argument if the function accepts it
            func_signature = inspect.signature(func)
            if 'verbose' in func_signature.parameters:
                resolved_args['verbose'] = args.verbose
            
            result = func(**resolved_args)
            
            if output_variable:
                local_vars[output_variable] = result
                local_vars["_current_df_"] = result # Update current_df if an output variable is set

            # Print result for relevant actions
            if function_name == "get_unique_values_for_canonical_feature":
                print(f"Unique values: {result}")
            elif isinstance(result, pl.DataFrame):
                print("Resulting DataFrame (first 5 rows):")
                print(result.head().to_pandas().to_markdown(index=False))
                print(f"Shape: {result.shape}")

        except Exception as e:
            print(f"Error executing step {step_idx + 1} ({function_name}): {e}")
            return

    print("Manipulation plan executed successfully.")
    # Save the final output if an output file is specified
    if output_variable and args.output_file:
        output_data = local_vars.get(output_variable)
        if output_data is not None:
            output_dir = os.path.dirname(args.output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            if isinstance(output_data, list):
                with open(args.output_file, 'w') as f:
                    for item in output_data:
                        f.write(f"{item}\n")
                print(f"Output saved to {args.output_file}")
            elif isinstance(output_data, pl.DataFrame):
                output_data.write_csv(args.output_file)
                print(f"Output DataFrame saved to {args.output_file}")
            else:
                print(f"Warning: Cannot save output of type {type(output_data)} to file.")

    return local_vars.get(output_variable) # Return the last outputted variable

def main():
    parser = argparse.ArgumentParser(description="Perform data manipulation based on a harmonization map.")
    parser.add_argument("harmonization_map_path", type=str,
                        help="Path to the JSON file containing the harmonization map (output from main.py).")
    parser.add_argument("data_folder_path", type=str,
                        help="Path to the folder containing the original datasets.")
    parser.add_argument("--action", type=str, required=True, choices=["unique_values", "merge", "filter", "llm_guided_manipulation", "generate_hypotheses"],
                        help="Action to perform: 'unique_values', 'merge', 'filter', 'llm_guided_manipulation', or 'generate_hypotheses'.")
    parser.add_argument("--canonical_feature", type=str,
                        help="The canonical name of the feature for 'unique_values' or 'filter' actions, or the merge key for 'merge' action.")
    parser.add_argument("--filter_value", type=str,
                        help="The value to filter by for the 'filter' action.")
    parser.add_argument("--request", type=str,
                        help="Natural language request for 'llm_guided_manipulation' action.")
    parser.add_argument("--output_file", type=str,
                        help="Optional: Path to save the output of the manipulation (e.g., unique values, merged data).")
    parser.add_argument("--llm_providers", type=str, default="google,nvidia,groq",
                        help="Comma-separated list of LLM providers to use, in order of preference (e.g., 'google,nvidia,groq').")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output for debugging.")

    args = parser.parse_args()

    harmonization_map = load_harmonization_map(args.harmonization_map_path)
    if harmonization_map is None:
        # Attempt to auto-generate harmonization map if not found
        print("Harmonization map not found. Attempting to auto-generate...")
        if not os.path.isdir(args.data_folder_path):
            print(f"Error: Data folder path '{args.data_folder_path}' is not a valid directory. Cannot auto-generate harmonization map.")
            return

        analysis_output_dir = os.path.join(args.data_folder_path, "Analysis")
        os.makedirs(analysis_output_dir, exist_ok=True)
        auto_generated_map_path = os.path.join(analysis_output_dir, "harmonization_map.json")

        all_analyses = {}
        print("---" + " Auto-generating Phase 1: Individual Dataset Analysis " + "---")
        for filename in os.listdir(args.data_folder_path):
            file_path = os.path.join(args.data_folder_path, filename)
            if os.path.isfile(file_path):
                print(f"Analyzing {filename}...")
                df_sample = read_dataset_sample(file_path)
                if df_sample is not None:
                    analysis = analyze_individual_dataset(file_path, df_sample, [p.strip() for p in args.llm_providers.split(',')])
                    if analysis:
                        all_analyses[filename] = analysis

        if not all_analyses:
            print("No datasets were successfully analyzed for auto-generation of harmonization map.")
            return

        print("\n---" + " Auto-generating Phase 2: Cross-Dataset Synthesis " + "---")
        harmonization_map = synthesize_analyses(all_analyses, "", [p.strip() for p in args.llm_providers.split(',')])

        if isinstance(harmonization_map, str):
            print(f"Error during auto-generation of synthesis: {harmonization_map}")
            return

        try:
            with open(auto_generated_map_path, 'w') as f:
                json.dump(harmonization_map, f, indent=2)
            print(f"\nAuto-generated harmonization map saved to: {auto_generated_map_path}")
            args.harmonization_map_path = auto_generated_map_path # Update path for subsequent use
        except Exception as e:
            print(f"Error saving auto-generated harmonization map to {auto_generated_map_path}: {e}")
            return

    if not os.path.isdir(args.data_folder_path):
        print(f"Error: The data folder path '{args.data_folder_path}' is not a valid directory.")
        return

    llm_providers_list = [p.strip() for p in args.llm_providers.split(',')]

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

    elif args.action == "llm_guided_manipulation":
        if not args.request:
            print("Error: --request is required for 'llm_guided_manipulation' action.")
            return
        plan_and_execute_manipulation(harmonization_map, args.data_folder_path, args.request, llm_providers_list, args)

    elif args.action == "generate_hypotheses":
        print("To generate hypotheses, please run 'python tools/hypothesis_generator.py' directly.")
        print("This script will prompt you for the necessary inputs.")

if __name__ == "__main__":
    main()