# Dataset Analyzer

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Dependencies](https://img.shields.io/badge/Dependencies-uv-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

The Dataset Analyzer is a powerful tool designed to streamline the process of understanding and harmonizing multiple datasets. Leveraging the advanced capabilities of various Large Language Models (LLMs), it performs deep semantic analysis on individual datasets and intelligently synthesizes these analyses to identify common features, suggest canonical names, and provide a comprehensive overview of your combined data landscape.

This tool is invaluable for researchers, data scientists, and analysts working with disparate datasets, enabling them to quickly identify relationships, standardize terminology, and unlock new insights from their data.

## Features

-   **Intelligent Semantic Analysis**: Utilizes various LLM APIs (Google Gemini, NVIDIA, Groq) to perform in-depth semantic understanding of each column within your datasets, describing its real-world meaning, data types, and suggesting potential synonyms.
-   **Automated Data Harmonization**: Automatically identifies and groups semantically identical columns across different datasets, facilitating consistent data integration.
-   **Canonical Naming Suggestions**: Proposes standardized, canonical names for grouped features, promoting clarity and uniformity in your data schema.
-   **Comprehensive Synthesis Report**: Generates a structured JSON report summarizing the combined information across all analyzed datasets, highlighting data relationships.
-   **Flexible Data Ingestion**: Supports common tabular data formats including CSV (`.csv`) and Excel (`.xlsx`, `.xls`) files.
-   **Robust LLM Fallback**: Implements a fallback mechanism to seamlessly switch between LLM providers (Google, NVIDIA, Groq) if one fails or hits rate limits, ensuring continuous operation.
-   **Configurable LLM Providers**: Allows users to specify the order of LLM providers to use via command-line arguments, offering flexibility and control.

## Getting Started

Follow these steps to set up and run the Dataset Analyzer on your local machine.

### Prerequisites

-   Python 3.9 or higher.
-   `uv` package manager (recommended for fast and reliable dependency management).

### Installation

1.  **Clone the Repository**

    First, clone the project repository to your local machine:

    ```bash
    git clone https://github.com/JJaniel/dataset_analyzer.git
    cd dataset_analyzer
    ```

2.  **Install `uv` (if not already installed)**

    If you don't have `uv` installed, you can get it via `pip`:

    ```bash
    pip install uv
    ```

3.  **Create and Activate Virtual Environment**

    It's highly recommended to use a virtual environment to manage project dependencies:

    ```bash
    uv venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

4.  **Install Project Dependencies**

    With your virtual environment activated, install the required packages:

    ```bash
    uv pip install -e .
    ```

5.  **Set Up LLM API Keys**

    The Dataset Analyzer uses various LLM APIs. You need to obtain API keys for the services you wish to use.

    -   **Google Gemini API**: Obtain from [Google AI Studio](https://aistudio.google.com/)
    -   **NVIDIA API**: Obtain from [NVIDIA AI Playground](https://build.nvidia.com/)
    -   **Groq API**: Obtain from [Groq Console](https://console.groq.com/)

    Once you have your API keys, create a file named `.env` in the root directory of this project and add your keys as follows:

    ```dotenv
    GOOGLE_API_KEY="your_google_api_key_here"
    NVIDIA_API_KEY="your_nvidia_api_key_here"
    GROQ_API_KEY="your_groq_api_key_here"
    ```

    Replace `"your_api_key_here"` with your actual API Keys.

## Usage

### 1. Generate Harmonization Map

To analyze your datasets and generate the harmonization map, run the `main.py` script. You can optionally save the output JSON to a file using the `--output_json` argument and specify the LLM providers to use.

```bash
python main.py <path_to_your_dataset_folder> [--output_json <output_file_path>] [--prompt "Your additional instructions here"] [--llm_providers "provider1,provider2,..."]
```

-   `<path_to_your_dataset_folder>`: The absolute or relative path to the folder containing your dataset files (CSV or Excel).
-   `--output_json <output_file_path>`: Optional. Path to save the harmonization map JSON output (e.g., `harmonization_map.json`).
-   `--prompt "Your additional instructions here"`: Optional. Additional prompt to include in the synthesis phase for custom requirements.
-   `--llm_providers "provider1,provider2,..."`: Optional. A comma-separated list of LLM providers to use, in order of preference. Supported providers are `google`, `nvidia`, and `groq`. If not specified, the default order is `google,nvidia,groq`.

**Examples:**

-   **Basic usage (using default LLM fallback order):**

    ```bash
    python main.py my_data
    ```

-   **Saving output to a JSON file:**

    ```bash
    python main.py my_data --output_json harmonization_map.json
    ```

-   **Using specific LLM providers (e.g., only NVIDIA and Groq, in that order):**

    ```bash
    python main.py my_data --llm_providers "nvidia,groq"
    ```

-   **Adding a custom prompt for synthesis:**

    ```bash
    python main.py my_data --prompt "Also, provide a brief summary of the most important findings."
    ```

### 2. Use Data Harmonizer for Manipulation

Once you have generated the `harmonization_map.json` file, you can use the `tools/data_harmonizer.py` script to perform various data manipulation tasks, such as extracting unique values for a canonical feature.

```bash
python tools/data_harmonizer.py <harmonization_map_path> <canonical_feature_name> <data_folder_path>
```

-   `<harmonization_map_path>`: Path to the JSON file containing the harmonization map (e.g., `harmonization_map.json`).
-   `<canonical_feature_name>`: The standardized name of the feature you want to analyze (e.g., `drug_id`, `cell_line_name`).
-   `<data_folder_path>`: Path to the folder containing your original datasets.

**Example:**

To get all unique drug IDs from your datasets using the generated `harmonization_map.json`:

```bash
python tools/data_harmonizer.py harmonization_map.json drug_id my_data
```

This will output a list of all unique values found for the `drug_id` canonical feature across all relevant datasets.

## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
