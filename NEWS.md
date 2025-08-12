# News

## Version 0.1.3

-   **Improved LLM Provider Selection**: Enhanced the LLM fallback mechanism to prioritize and stick with the first successful LLM provider for subsequent analyses, significantly improving efficiency and reducing redundant API calls.
-   **Increased LLM Token Limits**: Adjusted maximum token limits for NVIDIA and Groq LLMs to accommodate larger and more detailed harmonization map outputs, preventing truncation.
-   **Refined Harmonization Map Output**: Restructured the harmonization map JSON output for better clarity and conciseness. The output now features top-level `harmonization_details` (list of canonical features) and `dataset_info` (dataset-level metadata). Additionally, `original_columns` now includes detailed `data_type` and `nan_count` for each mapped column.

## Version 0.1.2

-   **Enhanced Individual Dataset Analysis**: Implemented sampling of unique values for each column to provide richer context for LLM analysis.
-   **Improved Data Manipulation Robustness**: Updated Polars merge operations to use `how='full'` for clearer and more robust data integration.
-   **Dependency Management**: Added `tabulate` as a required dependency for displaying merged data in markdown format.
-   **LLM Synthesis Stability**: Addressed and resolved issues causing LLM synthesis failures, including proper handling of dataset metadata and column name cleaning.

## Version 0.1.1

-   **Enhanced Metadata Integration**: Implemented automatic detection and utilization of metadata files (e.g., `.json`, `.txt`, `.yaml`, `.yml`, `.md`) located alongside datasets, providing richer context for LLM analysis.
-   **Improved LLM Output Robustness**: Enhanced JSON parsing logic to reliably handle varied LLM output formats, including responses wrapped in markdown code blocks.
-   **Circular Import Resolution**: Refactored code to eliminate circular dependencies, improving module structure and reliability.

## Version 0.1.0

- Initial release with dataset analysis and harmonization features.
- Added support for CSV and Excel (XLSX) files.
- Integrated with Google Gemini API for semantic analysis.