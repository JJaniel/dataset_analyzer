import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import json
from tools.llm_manager import get_llm_response

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
        return json.loads(cleaned_result)
    except Exception as e:
        print(f"An error occurred during individual analysis of {file_path}: {e}")
        return None
