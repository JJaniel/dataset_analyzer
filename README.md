# Dataset Analyzer

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Dependencies](https://img.shields.io/badge/Dependencies-uv-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

The Dataset Analyzer is a powerful tool designed to streamline the process of understanding and harmonizing multiple datasets. Leveraging the advanced capabilities of Google's Gemini model, it performs deep semantic analysis on individual datasets and intelligently synthesizes these analyses to identify common features, suggest canonical names, and provide a comprehensive overview of your combined data landscape.

This tool is invaluable for researchers, data scientists, and analysts working with disparate datasets, enabling them to quickly identify relationships, standardize terminology, and unlock new insights from their data.

## Features

-   **Intelligent Semantic Analysis**: Utilizes the Gemini API to perform in-depth semantic understanding of each column within your datasets, describing its real-world meaning, data types, and suggesting potential synonyms.
-   **Automated Data Harmonization**: Automatically identifies and groups semantically identical columns across different datasets, facilitating consistent data integration.
-   **Canonical Naming Suggestions**: Proposes standardized, canonical names for grouped features, promoting clarity and uniformity in your data schema.
-   **Comprehensive Synthesis Report**: Generates a human-readable report summarizing the combined information across all analyzed datasets, highlighting research potential and data relationships.
-   **Flexible Data Ingestion**: Supports common tabular data formats including CSV (`.csv`) and Excel (`.xlsx`, `.xls`) files.

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

5.  **Set Up Google API Key**

    The Dataset Analyzer uses the Google Gemini API for its core analysis. You need to obtain an API key from the Google AI Studio.

    -   Go to [Google AI Studio](https://aistudio.google.com/)
    -   Create a new API key.

    Once you have your API key, create a file named `.env` in the root directory of this project and add your key as follows:

    ```dotenv
    GOOGLE_API_KEY="your_google_api_key_here"
    ```

    Replace `"your_google_api_key_here"` with your actual Google API Key.

## Usage

### 1. Generate Harmonization Map

To analyze your datasets and generate the harmonization map, run the `main.py` script. You can optionally save the output JSON to a file using the `--output_json` argument.

```bash
python main.py <path_to_your_dataset_folder> [--output_json <output_file_path>] [--prompt "Your additional instructions here"]
```

**Example:**

If your datasets are located in a folder named `my_data` within the project directory, and you want to save the harmonization map to `harmonization_map.json`:

```bash
python main.py my_data --output_json harmonization_map.json
```

This will print the JSON harmonization map to the console and save it to the specified file.

### 2. Use Data Harmonizer for Manipulation

Once you have generated the `harmonization_map.json` file, you can use the `data_harmonizer.py` script to perform various data manipulation tasks, such as extracting unique values for a canonical feature.

```bash
python tools/data_harmonizer.py <harmonization_map_path> <canonical_feature_name> <data_folder_path>
```

-   `<harmonization_map_path>`: Path to the JSON file containing the harmonization map (e.g., `harmonization_map.json`).
-   `<canonical_feature_name>`: The standardized name of the feature you want to analyze (e.g., `drug_id`, `cell_line_name`).
-   `<data_folder_path>`: Path to the folder containing your original datasets.

**Example:**

To get all unique drug IDs from your datasets using the generated `harmonization_map.json`:

```bash
python data_harmonizer.py harmonization_map.json drug_id my_data
```

This will output a list of all unique values found for the `drug_id` canonical feature across all relevant datasets.

## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.