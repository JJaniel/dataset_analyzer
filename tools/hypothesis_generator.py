import json
from tools.llm_manager import get_llm_response
from langchain.prompts import PromptTemplate

def generate_hypotheses(harmonization_map_path: str, llm_providers: list[str], field_of_interest: str, web_context: str = ""):
    """
    Generates critical hypotheses based on the features available in the harmonization map,
    and relates them to a specific field of interest.

    Args:
        harmonization_map_path (str): The path to the harmonization map JSON file.
        llm_providers (list[str]): A list of LLM providers to use.
        field_of_interest (str): The specific field the user wants to relate the dataset to.
    """
    try:
        with open(harmonization_map_path, 'r') as f:
            harmonization_map = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{harmonization_map_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{harmonization_map_path}'.")
        return

    canonical_features = list(harmonization_map.keys())

    

    prompt_template = PromptTemplate(
        input_variables=["features", "field_of_interest", "web_context"],
        template="""
        You are an expert research scientist specializing in the field of {field_of_interest}.
        Your task is to analyze a dataset's harmonized features and generate critical, insightful hypotheses,
        relating them specifically to the field of {field_of_interest}. You should also provide
        relevant research context and findings from external sources.

        Here are the harmonized features from the dataset:
        {features}

        Here is some recent research and findings related to {field_of_interest} that might be relevant:
        {web_context}

        Please provide your analysis in the following structured format:

        --- Relation between Dataset and {field_of_interest} ---
        [Few concise points gathered from your research showing the relation between the dataset's features and the specified field.]

        --- Recent Research and Findings ---
        [Summarize key recent research and findings from the provided web context that support or inform your hypotheses,
        or highlight relevant trends in {field_of_interest} related to data analysis.]

        --- Possible Hypotheses ---
        [A list of 3-5 testable hypotheses. For each, provide a brief explanation of the reasoning,
        ensuring they are specific, measurable, and relevant to both the dataset features and {field_of_interest}.]
        """
    )

    print("Generating hypotheses and insights...")
    hypotheses, _ = get_llm_response(
        prompt_template,
        {
            "features": ", ".join(canonical_features),
            "field_of_interest": field_of_interest,
            "web_context": web_context
        },
        llm_providers
    )

    if hypotheses:
        print("\n--- Generated Hypotheses and Insights ---")
        print(hypotheses)
    else:
        print("Could not generate hypotheses and insights. The LLM did not return a response.")


if __name__ == "__main__":
    harmonization_map_path = input("Enter the path to the harmonization map JSON file: ").strip()

    field_of_interest = input("Which Field You want to relate to this dataset (e.g., agriculture, Biomedical, Automobile): ").strip()

    llm_providers_input = input("Enter comma-separated LLM providers (e.g., google,nvidia,groq) or press Enter for default (google,nvidia,groq): ").strip()
    if not llm_providers_input:
        llm_providers = ["google", "nvidia", "groq"]
    else:
        llm_providers = [p.strip() for p in llm_providers_input.split(',')]

    generate_hypotheses(harmonization_map_path, llm_providers, field_of_interest)
