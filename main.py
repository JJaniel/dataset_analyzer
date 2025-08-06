import os
import argparse
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
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

def analyze_individual_dataset(file_path, df):
    """
    Analyzes a single dataset to understand its structure and semantic meaning.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
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

    prompt = PromptTemplate(template=template, input_variables=["file_name", "dataset_sample"])
    chain = prompt | llm

    try:
        result = chain.invoke({
            "file_name": os.path.basename(file_path),
            "dataset_sample": df.to_string()
        })
        # Clean the output to ensure it's valid JSON
        cleaned_result = result.content.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_result)
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
    args = parser.parse_args()

    folder_path = args.folder_path
    additional_prompt = args.prompt
    output_json_path = args.output_json

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
                analysis = analyze_individual_dataset(file_path, df)
                if analysis:
                    all_analyses[filename] = analysis

    if not all_analyses:
        print("No datasets were successfully analyzed.")
        return

    print("\n--- Phase 2: Cross-Dataset Synthesis ---")
    print("Synthesizing results to find common features...")
    harmonization_map = synthesize_analyses(all_analyses, additional_prompt)

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