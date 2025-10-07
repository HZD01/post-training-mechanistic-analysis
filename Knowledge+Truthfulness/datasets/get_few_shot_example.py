import pandas as pd
import glob
import random
import json
import os

def read_and_sample_csvs(csv_folder_path, seed=42, output_file="few_shot_samples.json"):

    random.seed(seed)
    

    csv_files = glob.glob(f"{csv_folder_path}/*.csv")
    datasets = []

    for file in csv_files:

        df = pd.read_csv(file)
        dataset_name = os.path.basename(file).replace(".csv", "")
        
        if 'statement' not in df.columns or 'label' not in df.columns:
            print(f"Skipping {file}, required columns not found.")
            continue
        
        label_0_samples = df[df['label'] == 0].sample(n=min(2, len(df[df['label'] == 0])), random_state=seed)
        label_1_samples = df[df['label'] == 1].sample(n=min(2, len(df[df['label'] == 1])), random_state=seed)
        
        examples = []
        for _, row in pd.concat([label_0_samples, label_1_samples]).iterrows():
            label_text = "TRUE" if row['label'] == 1 else "FALSE"
            sample_text = f"{row['statement']} This statement is: {label_text}"
            examples.append(sample_text)
        
        datasets.append({
            "dataset_name": dataset_name,
            "few_shot_examples": examples
        })
    

    with open(output_file, "w") as json_file:
        json.dump(datasets, json_file, indent=4)
    
    print(f"Few-shot examples saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    csv_folder = "."
    seed_value = 1
    output_filename = "few_shot_samples.json"
    read_and_sample_csvs(csv_folder, seed=seed_value, output_file=output_filename)