import os
import argparse
import pandas as pd
from langchain.prompts import PromptTemplate
from tools.llm_manager import get_llm_response
from dotenv import load_dotenv
import json
from tools.data_synthesizer import synthesize_analyses
from tools.data_analyzer import analyze_individual_dataset
from tools.utils import parse_json_with_fix, read_dataset_sample

def main():
    """
    The main function to run the multi-dataset analysis.
    """
    parser = argparse.ArgumentParser(description="Analyze and harmonize multiple datasets in a folder.")
    parser.add_argument("folder_path", type=str, help="The path to the folder containing your datasets.")
    parser.add_argument("--prompt", type=str, default="",
                        help="Additional prompt to include in the synthesis phase for custom requirements.")
    parser.add_argument("--output_json", type=str, default="harmonization_map.json",
                        help="Optional: Path to save the harmonization map JSON output.")
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
    successful_provider = None
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            print(f"Analyzing {filename}...")
            df_sample = read_dataset_sample(file_path)
            if df_sample is not None:
                metadata_content = None
                base_filename, _ = os.path.splitext(filename)
                METADATA_EXTENSIONS = ['.json', '.txt', '.yaml', '.yml', '.md'] # Common metadata extensions

                for ext in METADATA_EXTENSIONS:
                    metadata_filename = base_filename + ext
                    metadata_file_path = os.path.join(folder_path, metadata_filename)
                    if os.path.isfile(metadata_file_path):
                        try:
                            with open(metadata_file_path, 'r', encoding='utf-8') as f:
                                metadata_content = f.read()
                            print(f"  Found metadata file: {metadata_filename}")
                            break # Found metadata, no need to check other extensions
                        except Exception as e:
                            print(f"  Error reading metadata file {metadata_filename}: {e}")

                providers_to_try = [successful_provider] if successful_provider else llm_providers
                analysis, provider = analyze_individual_dataset(file_path, df_sample, providers_to_try, metadata_content)
                if analysis:
                    all_analyses[filename] = analysis
                    if provider:
                        successful_provider = provider
                        # Also update the main list to prioritize the successful one for the synthesis phase
                        if provider in llm_providers:
                            llm_providers.remove(provider)
                            llm_providers.insert(0, provider)


    if not all_analyses:
        print("No datasets were successfully analyzed.")
        return

    print("\n--- Phase 2: Cross-Dataset Synthesis ---")
    print("Synthesizing results to find common features...")
    # The llm_providers list is now prioritized with the successful provider
    harmonization_map = synthesize_analyses(all_analyses, additional_prompt, llm_providers)

    if isinstance(harmonization_map, str):
        print(f"Error during synthesis: {harmonization_map}")
        return

    print("\n--- Harmonization Map (JSON) ---")
    print(json.dumps(harmonization_map, indent=2))

    if output_json_path:
        output_dir = os.path.dirname(output_json_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            with open(output_json_path, 'w') as f:
                json.dump(harmonization_map, f, indent=2)
            print(f"\nHarmonization map saved to: {output_json_path}")
        except Exception as e:
            print(f"Error saving harmonization map to {output_json_path}: {e}")


if __name__ == "__main__":
    main()