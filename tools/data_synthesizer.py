import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

def synthesize_analyses(all_analyses, additional_prompt=""):
    """
    Synthesizes analyses from multiple datasets to find common features.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    template = """
    You are a research assistant. Your goal is to harmonize multiple dataset analyses to help a researcher understand how their data fits together.

    Here are the individual analyses from multiple dataset files:
    {analyses_json}

    Based on the provided analyses, perform the following tasks and provide the output in JSON format:
    1.  **Identify Feature Groups**: Group together column names from different files that you believe are semantically identical (i.e., they represent the same real-world feature).
    2.  **Suggest Canonical Names**: For each group, suggest a single, standardized "canonical" name.
    3.  **Provide Semantic Meaning and Data Type**: For each canonical feature, provide a brief semantic meaning and its likely data type.
    4.  **Map Original Columns**: For each canonical feature, list the original column names from each dataset file that map to it.

    The JSON output should be a list of objects, where each object represents a canonical feature. Each object should have the following keys:
    -   `canonical_name`: The suggested standardized name for the feature.
    -   `semantic_meaning`: A brief description of what the feature represents.
    -   `data_type`: The likely data type of the feature (e.g., "Integer", "String", "Float").
    -   `original_columns`: An object where keys are dataset filenames and values are lists of original column names from that dataset that map to this canonical feature.

    Example JSON structure:
    ```json
    [
      {{
        "canonical_name": "drug_id",
        "semantic_meaning": "Unique identifier for a drug.",
        "data_type": "Integer",
        "original_columns": {{
          "dataset1.csv": ["DRUG_ID_1", "DRUG_ID_A"],
          "dataset2.xlsx": ["DrugID"]
        }}
      }},
      {{
        "canonical_name": "patient_age",
        "semantic_meaning": "Age of the patient.",
        "data_type": "Integer",
        "original_columns": {{
          "dataset1.csv": ["Age"],
          "dataset3.csv": ["PatientAge", "Age_Years"]
        }}
      }}
    ]
    ```

    Provide only the JSON output.

    {additional_prompt}
    """

    prompt = PromptTemplate(template=template, input_variables=["analyses_json", "additional_prompt"])
    chain = prompt | llm

    try:
        analyses_str = json.dumps(all_analyses, indent=2)
        result = chain.invoke({"analyses_json": analyses_str, "additional_prompt": additional_prompt})
        cleaned_result = result.content.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_result)
    except Exception as e:
        return f"An error occurred during synthesis: {e}"
