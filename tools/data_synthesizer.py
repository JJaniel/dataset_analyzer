import json
from langchain.prompts import PromptTemplate
from tools.llm_manager import get_llm_response

def synthesize_analyses(all_analyses, additional_prompt="", llm_providers=None):
    """
    Synthesizes analyses from multiple datasets to find common features.
    """
    template = """
    You are a research assistant. Your goal is to harmonize multiple dataset analyses to help a researcher understand how their data fits together.

    Here are the individual analyses from multiple dataset files:
    {analyses_json}

    Based on the provided analyses, perform the following tasks and provide the output in a single JSON object.

    The JSON object should have two top-level keys: "harmonization_details" and "dataset_info".

    1.  **`harmonization_details`**: This should be a list of objects, where each object represents a canonical feature. Each object should have the following keys:
        -   `canonical_name`: The suggested standardized name for the feature.
        -   `semantic_meaning`: A brief description of what the feature represents.
        -   `data_type`: The likely data type of the feature (e.g., "Integer", "String", "Float").
        -   `original_columns`: An object where keys are dataset filenames and values are lists of objects. Each inner object should contain:
            -   `column_name`: The original column name from the dataset.
            -   `data_type`: The data type of the original column.
            -   `nan_count`: The number of NaN/Null values in the original column.

    2.  **`dataset_info`**: This should be an object where keys are the dataset filenames. Each value should be an object containing the dataset's metadata, such as its `shape`.

    Example JSON structure:
    ```json
    {{
      "harmonization_details": [
        {{
          "canonical_name": "drug_id",
          "semantic_meaning": "Unique identifier for a drug.",
          "data_type": "Integer",
          "original_columns": {{
            "dataset1.csv": [
              {{
                "column_name": "DRUG_ID_1",
                "data_type": "Integer",
                "nan_count": 0
              }},
              {{
                "column_name": "DRUG_ID_A",
                "data_type": "Integer",
                "nan_count": 5
              }}
            ],
            "dataset2.xlsx": [
              {{
                "column_name": "DrugID",
                "data_type": "Integer",
                "nan_count": 2
              }}
            ]
          }}
        }}
      ],
      "dataset_info": {{
        "dataset1.csv": {{
          "shape": [100, 5]
        }},
        "dataset2.xlsx": {{
          "shape": [150, 7]
        }}
      }}
    }}
    ```

    Provide only the JSON output.

    {additional_prompt}
    """

    # No need to create dataset_metadata separately, the LLM will do it.
    input_variables = {
        "analyses_json": json.dumps(all_analyses, indent=2),
        "additional_prompt": additional_prompt
    }

    try:
        result_content, _ = get_llm_response(template, input_variables, llm_providers)
        cleaned_result = result_content.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_result)
    except Exception as e:
        return f"An error occurred during synthesis: {e}"