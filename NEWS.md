# News

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