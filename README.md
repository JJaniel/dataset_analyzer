# Dataset Analyzer

This project provides a tool to analyze and harmonize multiple datasets within a specified folder. It uses Google's Gemini model to perform semantic analysis on individual datasets and then synthesizes the results to identify common features and suggest canonical names.

## Features

- **Individual Dataset Analysis**: Analyzes each dataset file to understand its structure, semantic meaning of columns, data types, and potential synonyms.
- **Cross-Dataset Synthesis**: Harmonizes analyses from multiple datasets to find common features and suggest standardized canonical names.
- **Human-Readable Report**: Generates a clear, final report summarizing the combined information across all datasets.

## Setup

1.  **Clone the repository (if you haven't already):**

    ```bash
    git clone <repository_url>
    cd dataset_analyzer
    ```

2.  **Install `uv` (if you don't have it):**

    ```bash
    pip install uv
    ```

3.  **Install dependencies using `uv`:**

    ```bash
    uv pip install -e .
    ```

    This will install all necessary packages listed in `pyproject.toml`.

4.  **Set up your Google API Key:**

    Create a `.env` file in the root directory of the project and add your Google API key:

    ```
    GOOGLE_API_KEY="your_google_api_key_here"
    ```

    Replace `"your_google_api_key_here"` with your actual Google API Key.

## Usage

To run the dataset analyzer, execute the `main.py` script with the path to your folder containing the datasets:

```bash
python main.py <path_to_your_dataset_folder>
```

**Example:**

If your datasets are in a folder named `my_data` in the same directory as `main.py`:

```bash
python main.py my_data
```

The script will output a final report to the console, detailing the semantic analysis and synthesis of your datasets.
