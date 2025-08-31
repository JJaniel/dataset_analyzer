from tools.llm_manager import get_llm_response
from tools.utils import parse_json_with_fix
import json

def synthesize_analyses(all_analyses, additional_prompt, llm_providers):
    """
    Synthesizes individual dataset analyses to create a cross-dataset harmonization map.
    """
    # Prepare the combined analyses for the LLM prompt
    analyses_json_str = json.dumps(all_analyses, indent=2)

    # Create the prompt for the LLM
    template = f"""
    You are a data harmonization expert. Your task is to analyze a collection of individual dataset analyses and create a single, unified "harmonization map".

    Here are the analyses of multiple datasets:
    {{analyses}}

    Your goal is to identify columns across these different datasets that represent the same underlying feature, even if they have different names.

    The output should be a JSON object where:
    - Each key is a "canonical_feature_name" that you create to represent a common concept (e.g., "patient_id", "tumor_size_mm").
    - The value for each key is an object containing:
      - "description": A brief explanation of what this canonical feature represents.
      - "mapped_columns": A list of objects, where each object details a specific column from a dataset that maps to this canonical feature.
        - "dataset": The name of the dataset file.
        - "column": The original name of the column in that dataset.
        - "semantic_meaning": The original semantic meaning of that column.

    {additional_prompt}

    Please provide only the final JSON output for the harmonization map.
    """

    input_variables = {"analyses": analyses_json_str}

    try:
        result_content, _ = get_llm_response(template, input_variables, llm_providers)
        cleaned_result = result_content.strip().replace("```json", "").replace("```", "")
        return parse_json_with_fix(cleaned_result)
    except Exception as e:
        return f"An error occurred during synthesis: {e}"
