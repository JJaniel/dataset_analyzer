import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import json
from tools.llm_manager import get_llm_response
from tools.utils import parse_json_with_fix

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

def analyze_individual_dataset(file_path, df, llm_providers, metadata_content=None):
    """
    Analyzes a single dataset to understand its structure and semantic meaning.
    """
    metadata_instruction = ""
    if metadata_content:
        metadata_instruction = f"\n\nAdditional Metadata (if available and relevant):\n{{metadata_content}}"

    template = f"""
    You are a data analyst. Your task is to perform a deep semantic analysis of the following dataset sample.

    Dataset File: {{file_name}}
    Sample Data:
    {{dataset_sample}}
    {metadata_instruction}

    Your analysis should be in JSON format and include:
    1.  **semantic_meaning**: For each column, describe what it likely represents in the real world.
    2.  **data_types_and_content**: Briefly describe the data type and content of each column.
    3.  **potential_synonyms**: Suggest alternative names for columns that might appear in other datasets.

    Provide only the JSON output, with no additional text or markdown formatting.
    """

    input_variables = {
        "file_name": os.path.basename(file_path),
        "dataset_sample": df.to_string()
    }
    if metadata_content:
        input_variables["metadata_content"] = metadata_content


    try:
        result_content = get_llm_response(template, input_variables, llm_providers)
        print(f"Raw LLM response: {result_content}") # Debugging line
        cleaned_result = result_content.strip().replace("```json", "").replace("```", "")
        # Use the robust JSON parser from main.py
        return parse_json_with_fix(cleaned_result)
    except Exception as e:
        print(f"An error occurred during individual analysis of {file_path}: {e}")
        return None
