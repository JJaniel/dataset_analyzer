import os
import argparse
import pandas as pd
from langchain.prompts import PromptTemplate
from tools.llm_manager import get_llm_response
from dotenv import load_dotenv
import json
from tools.data_synthesizer import synthesize_analyses

load_dotenv()

def get_dataset_sample(file_path):
    """
    Reads the first 3 rows of a dataset file.
    Note: This reads the *first* 3 rows. For truly random rows from large files,
    a more advanced sampling strategy would be required (e.g., reading chunks).
    """
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, nrows=3)
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path, nrows=3)
        else:
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

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
                    json_string += ']' # Try to close an array
                elif json_string.endswith(']'):
                    json_string += '}' # Try to close an object
                else:
                    json_string += '}' # Default to closing an object
            elif "Expecting property name enclosed in double quotes" in str(e):
                # This is harder to fix automatically without more context
                pass
            
            if i < retries - 1:
                print("Attempting to fix JSON and retry...")
            else:
                raise # Re-raise if all retries fail

    return json.loads(json_string) # Should not be reached if retries fail

def analyze_individual_dataset(file_path, df, llm_providers):
    """
    Analyzes a single dataset to understand its structure and semantic meaning.
    """
    template = """
    You are a data analyst. Your task is to perform a deep semantic analysis of the following dataset sample.

    Dataset File: {file_name}
    Sample Data:
    {dataset_sample}

    Your analysis should be in JSON format and include:
    1.  **semantic_meaning**: For each column, describe what it likely represents in the real world.
    2.  **data_types_and_content**: Briefly describe the data type and content of each column.
    3.  **potential_synonyms**: Suggest alternative names for columns that might appear in other datasets.

    Example Output:
    {{
        "col1": {{
            "semantic_meaning": "A unique identifier for a user.",
            "data_types_and_content": "Integer.",
            "potential_synonyms": ["user_id", "customer_id"]
        }},
        "col2": {{
            "semantic_meaning": "The age of the user.",
            "data_types_and_content": "Integer.",
            "potential_synonyms": ["age", "user_age"]
        }}
    }}

    Provide only the JSON output.
    """

    input_variables = {
        "file_name": os.path.basename(file_path),
        "dataset_sample": df.to_string()
    }

    try:
        result_content = get_llm_response(template, input_variables, llm_providers)
        # Clean the output to ensure it's valid JSON
        cleaned_result = result_content.strip().replace("```json", "").replace("```", "")
        return parse_json_with_fix(cleaned_result)
    except Exception as e:
        print(f"An error occurred during individual analysis of {file_path}: {e}")
        return None

def main():
    """
    The main function to run the multi-dataset analysis.
    """
    parser = argparse.ArgumentParser(description="Analyze and harmonize multiple datasets in a folder.")
    parser.add_argument("folder_path", type=str, help="The path to the folder containing your datasets.")
    parser.add_argument("--prompt", type=str, default="",
                        help="Additional prompt to include in the synthesis phase for custom requirements.")
    parser.add_argument("--output_json", type=str, help="Optional: Path to save the harmonization map JSON output.")
    parser.add_argument("--llm_providers", type=str, default="google,nvidia,groq",
                        help="Comma-separated list of LLM providers to use, in order of preference (e.g., 'google,nvidia,groq').")
    args = parser.parse_args()

    folder_path = args.folder_path
    additional_prompt = args.prompt
    output_json_path = args.output_json
    llm_providers = [p.strip() for p in args.llm_providers.split(',')]

    if not os.path.isdir(folder_path):
        print(f"Error: The path '{folder_path}' is not a valid directory.")
        return

    all_analyses = {}
    print("--- Phase 1: Individual Dataset Analysis ---")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            print(f"Analyzing {filename}...")
            df = get_dataset_sample(file_path)
            if df is not None:
                analysis = analyze_individual_dataset(file_path, df, llm_providers)
                if analysis:
                    all_analyses[filename] = analysis

    if not all_analyses:
        print("No datasets were successfully analyzed.")
        return

    print("\n--- Phase 2: Cross-Dataset Synthesis ---")
    print("Synthesizing results to find common features...")
    harmonization_map = synthesize_analyses(all_analyses, additional_prompt, llm_providers)

    if isinstance(harmonization_map, str):
        print(f"Error during synthesis: {harmonization_map}")
        return

    print("\n--- Harmonization Map (JSON) ---")
    print(json.dumps(harmonization_map, indent=2))

    if output_json_path:
        try:
            with open(output_json_path, 'w') as f:
                json.dump(harmonization_map, f, indent=2)
            print(f"\nHarmonization map saved to: {output_json_path}")
        except Exception as e:
            print(f"Error saving harmonization map to {output_json_path}: {e}")

if __name__ == "__main__":
    main()
