
import os
import argparse
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import json

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

def synthesize_analyses(all_analyses):
    """
    Synthesizes analyses from multiple datasets to find common features.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    
    template = """
    You are a research assistant. Your goal is to harmonize multiple dataset analyses to help a researcher understand how their data fits together.

    Here are the individual analyses from multiple dataset files:
    {analyses_json}

    Based on the provided analyses, perform the following tasks and provide the output in a clear, final report format:
    1.  **Identify Feature Groups**: Group together column names from different files that you believe are semantically identical (i.e., they represent the same real-world feature).
    2.  **Suggest Canonical Names**: For each group, suggest a single, standardized "canonical" name.
    3.  **Provide an Overall Summary**: Briefly describe the combined information available across all datasets and suggest what kind of research might be possible with this combined data.

    Do not output JSON. Format the output as a human-readable report.
    """

    prompt = PromptTemplate(template=template, input_variables=["analyses_json"])
    chain = prompt | llm

    try:
        analyses_str = json.dumps(all_analyses, indent=2)
        result = chain.invoke({"analyses_json": analyses_str})
        return result.content
    except Exception as e:
        return f"An error occurred during synthesis: {e}"

def main():
    """The main function to run the multi-dataset analysis."""
    parser = argparse.ArgumentParser(description="Analyze and harmonize multiple datasets in a folder.")
    parser.add_argument("folder_path", type=str, help="The path to the folder containing your datasets.")
    args = parser.parse_args()

    folder_path = args.folder_path
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
    final_report = synthesize_analyses(all_analyses)

    print("\n--- Final Report ---")
    print(final_report)

if __name__ == "__main__":
    main()
