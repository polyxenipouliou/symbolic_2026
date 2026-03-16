import pandas as pd
import os

# 1. Define file paths
file_paths = {
    "bert": r"E:\Master\symbolic_2026\Adversarial-MidiBERT\midibert_768d_features.csv",
    "features": r"E:\Master\symbolic_2026\handmade_54_features.csv",
    "stats": r"E:\Master\symbolic_2026\features\filtered_features_statistical.csv"
}

# Define output path
output_path = r"E:\Master\symbolic_2026\feature_12+54+768.csv"


def merge_music_data(files, save_to):
    print("Reading and merging data...")

    # Read the first file as the base table
    # Assumes each file contains at least 'filename' and 'composer' columns
    df_bert = pd.read_csv(files["bert"])
    df_feat = pd.read_csv(files["features"])
    df_stat = pd.read_csv(files["stats"])

    # 2. Execute merge logic (use inner join to keep only filenames present in all datasets)
    # Merge based on both 'filename' and 'composer' to avoid duplicate composer columns
    merged_df = pd.merge(df_bert, df_feat, on=["filename", "composer"], how="inner")

    # Then merge with stats
    final_df = pd.merge(merged_df, df_stat, on=["filename", "composer"], how="inner")

    # 3. Apply filtering logic: Exclude data from "Clara Schumann"
    if "composer" in final_df.columns:
        initial_count = len(final_df)
        final_df = final_df[final_df["composer"] != "Clara Schumann"]
        removed_count = initial_count - len(final_df)
        if removed_count > 0:
            print(f"Excluded {removed_count} entries by Clara Schumann")

    # 4. Save the result
    final_df.to_csv(save_to, index=False)

    print("-" * 30)
    print("Merge successful!")
    print(f"Total number of features (columns): {final_df.shape[1]}")
    print(f"Total number of samples (rows): {final_df.shape[0]}")
    print(f"File saved to: {save_to}")


if __name__ == "__main__":
    try:
        merge_music_data(file_paths, output_path)
    except KeyError as e:
        print(f"Error: Missing required columns (e.g., 'filename' or 'composer') in some files: {e}")
    except Exception as e:
        print(f"Runtime error: {e}")