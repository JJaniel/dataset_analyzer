# Commands
python main.py <folder> --output_json harmonization_map.json
python tools/data_manipulator.py harmonization_map.json <folder> --action merge --canonical_feature <feature>
python tools/hypothesis_generator.py

# Hypothesis Generation Procedure
To generate hypotheses, run `python tools/hypothesis_generator.py` interactively. The script will prompt for:
- Path to the harmonization map JSON file.
- Field of interest (e.g., agriculture, Biomedical, Automobile).
- Comma-separated LLM providers (e.g., google,nvidia,groq). `nvidia_nemotron` is recommended for this task.

The output will be structured into:
- Relation between Dataset and [Field of Interest]
- Recent Research and Findings
- Possible Hypotheses

# Style
- Use descriptive variable names: df, file_path, llm_providers
- Imports: stdlib → 3rd party → local (see main.py)
- JSON handling: always strip ```json``` and ``` markers
- Error handling: print(f"Error... {e}") and return None
- Function docs: triple quotes, brief imperative sentences
- Type hints: minimal (based on codebase)
- 120 char limit (observed in codebase)

# Testing
No test framework installed. Manual testing via:
python main.py test_data --llm_providers "google,groq"

# LLM Providers
GOOGLE_API_KEY, NVIDIA_API_KEY, GROQ_API_KEY in .env