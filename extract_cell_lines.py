import pandas as pd
import sys
import os

input_file_paths = sys.argv[1:-1]
output_csv_path = sys.argv[-1]

all_unique_cell_lines = set()

# Read existing unique cell lines if the output file already exists
if os.path.exists(output_csv_path):
    try:
        existing_df = pd.read_csv(output_csv_path)
        if 'CELL_LINE' in existing_df.columns:
            all_unique_cell_lines.update(existing_df['CELL_LINE'].unique())
    except Exception as e:
        print(f"Warning: Could not read existing unique cell lines from {output_csv_path}: {e}")

for input_path in input_file_paths:
    try:
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        elif input_path.endswith('.xlsx'):
            df = pd.read_excel(input_path)
        else:
            print(f"Skipping unsupported file type: {input_path}")
            continue

        if 'CELL_LINE' in df.columns:
            all_unique_cell_lines.update(df['CELL_LINE'].unique())
        else:
            print(f"Error: 'CELL_LINE' column not found in {input_path}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
    except Exception as e:
        print(f"An error occurred while processing {input_path}: {e}")

if all_unique_cell_lines:
    pd.DataFrame(sorted(list(all_unique_cell_lines)), columns=['CELL_LINE']).to_csv(output_csv_path, index=False)
    print(f"Successfully extracted and consolidated unique CELL_LINEs to {output_csv_path}")
else:
    print("No unique CELL_LINEs found or processed.")