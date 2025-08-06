import os
import argparse
import pandas as pd
from langchain.prompts import PromptTemplate
from tools.llm_manager import get_llm_response
from dotenv import load_dotenv
import json
from tools.data_synthesizer import synthesize_analyses
from tools.data_analyzer import analyze_individual_dataset, get_dataset_sample

load_dotenv()

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
                    json_string += ']'
                elif json_string.endswith(']'):
                    json_string += '}'
                else:
                    json_string += '}'
            elif "Expecting property name enclosed in double quotes" in str(e):
                # This is harder to fix automatically without more context
                pass
            
            if i < retries - 1:
                print("Attempting to fix JSON and retry...")
            else:
                raise # Re-raise if all retries fail

    return json.loads(json_string) # Should not be reached if retries fail

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