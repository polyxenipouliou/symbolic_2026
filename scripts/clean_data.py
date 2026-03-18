import pandas as pd
import os

# 1. Define file paths (using r"" to handle escape characters correctly)
file_names = {
    "bert": r"E:\Master\symbolic_2026\Adversarial-MidiBERT\midibert_768d_features.csv",
    "features": r"E:\Master\symbolic_2026\handmade_60_features.csv",
    "stats": r"E:\Master\symbolic_2026\features\features_statistical.csv",  # Please verify if these filenames are correct
    "bars": r"E:\Master\symbolic_2026\features\features_sequential.csv"
}


def process_music_data(files):
    # Read all CSV files into a dictionary of DataFrames
    dfs = {k: pd.read_csv(v) for k, v in files.items()}

    # 2. Find the intersection of 'filename' columns across all datasets
    # Start with the set of filenames from the first dataset ('bert')
    common_filenames = set(dfs["bert"]["filename"])

    # Intersect with filenames from all other datasets
    for df in dfs.values():
        common_filenames &= set(df["filename"])

    print(f"Number of common files found: {len(common_filenames)}")

    # 3. Filter data and save results
    for key, df in dfs.items():
        # Filter the DataFrame to keep only rows with common filenames
        filtered_df = df[df["filename"].isin(common_filenames)].copy()

        # Exclude data composed by "Clara Schumann" if the column exists
        if "composer" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["composer"] != "Clara Schumann"]

        # 4. Construct the output file path correctly
        original_path = files[key]
        directory = os.path.dirname(original_path)  # Get the directory path
        filename = os.path.basename(original_path)  # Get the original filename

        # Create new filename with 'filtered_' prefix
        output_name = os.path.join(directory, f"filtered_{filename}")

        # Save the filtered DataFrame to CSV
        filtered_df.to_csv(output_name, index=False)
        print(f"Generated: {output_name} (Remaining rows: {len(filtered_df)})")


if __name__ == "__main__":
    try:
        process_music_data(file_names)
        print("\nAll files processed successfully!")
    except Exception as e:
        print(f"Error occurred during execution: {e}")