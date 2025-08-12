import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import json
from tools.llm_manager import get_llm_response
from tools.utils import parse_json_with_fix, get_unique_values_sample, read_dataset_sample, read_full_dataset

def analyze_individual_dataset(file_path, df_sample, llm_providers, metadata_content=None):
    """
    Analyzes a single dataset to understand its structure and semantic meaning.
    df_sample: A sample of the dataframe for LLM analysis.
    Returns the analysis and the name of the successful provider.
    """
    metadata_instruction = ""
    if metadata_content:
        metadata_instruction = f"\n\nAdditional Metadata (if available and relevant):\n{{metadata_content}}"

    unique_values_sample = get_unique_values_sample(df_sample)

    # Read the full dataset to get accurate shape and NaN counts
    df_full = read_full_dataset(file_path)
    if df_full is None:
        print(f"Error: Could not read full dataset from {file_path} for shape and NaN counts.")
        return None, None

    # Clean column names by stripping whitespace
    df_full.columns = df_full.columns.str.strip()

    dataset_shape = list(df_full.shape)
    nan_null_counts = df_full.isnull().sum().to_dict()

    template = f"""
    You are a data analyst. Your task is to perform a deep semantic analysis of the following dataset sample.

    Dataset File: {{file_name}}
    Sample Data (JSON format):
    {{dataset_sample}}

    Sample of Unique Values per Column:
    {{unique_values_sample}}

    Dataset Shape (rows, columns): {{dataset_shape}}
    NaN/Null Counts per Column: {{nan_null_counts}}
    {metadata_instruction}

    Your analysis should be in JSON format and include:
    1.  **semantic_meaning**: For each column, describe what it likely represents in the real world.
    2.  **data_types_and_content**: Briefly describe the data type and content of each column.
    3.  **potential_synonyms**: Suggest alternative names for columns that might appear in other datasets.
    4.  **shape**: The shape of the dataset (rows, columns).
    5.  **nan_null_counts**: A dictionary of NaN/Null counts per column.

    Provide only the JSON output, with no additional text or markdown formatting.
    """

    input_variables = {
        "file_name": os.path.basename(file_path),
        "dataset_sample": df_sample.to_json(orient='records', indent=2),
        "unique_values_sample": json.dumps(unique_values_sample, indent=2),
        "dataset_shape": dataset_shape,
        "nan_null_counts": json.dumps(nan_null_counts, indent=2)
    }
    if metadata_content:
        input_variables["metadata_content"] = metadata_content


    try:
        result_content, successful_provider = get_llm_response(template, input_variables, llm_providers)
        print(f"Raw LLM response: {result_content}") # Debugging line
        cleaned_result = result_content.strip().replace("```json", "").replace("```", "")
        # Use the robust JSON parser from main.py
        return parse_json_with_fix(cleaned_result), successful_provider
    except Exception as e:
        print(f"An error occurred during individual analysis of {file_path}: {e}")
        return None, None

