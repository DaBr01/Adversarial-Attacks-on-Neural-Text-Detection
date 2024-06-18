import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

# change experiment folder name to do visuals for other formality family tests.
df = pd.read_csv("../../data/Various tests/results/result.csv")
save_folder = "..\..\data\Various tests\\results"

df["mutation"] = df["text_id"].apply(lambda x: x.split("_")[0])
char_mut_names = {
    "def": "Baseline",
    "ref": "Reference list and citations",
    "cit": "Citations no reference list",
    "num": "Numerical data",
    "gram": "Unusual grammar&sentence structures",
    "form": "Formality blend",
    "first": "First person"
}


print(df.head())
# Getting unique detectors
detectors = df['detector'].unique()

df['score'] = pd.to_numeric(df['score'], errors='coerce')
print(df.head())
# check if empty scores
empty_score_text_ids = df[pd.isna(df['score'])]['text_id']
print(empty_score_text_ids)

df['mutation_name'] = df['mutation'].map(char_mut_names)
print(df.head())
df.to_csv(os.path.join(save_folder, "simulating_features_of_outlying_texts.csv"))
# Step 3: Group by 'detector' and 'mutation_name', then calculate average scores
avg_scores = df.groupby(['detector', 'mutation_name'])['score'].mean().reset_index()

# Now, plot the histograms
unique_detectors = avg_scores['detector'].unique()
for detector in unique_detectors:
    detector_data = avg_scores[avg_scores['detector'] == detector]
    plt.figure(figsize=(10, 6))  # Adjusted for a taller and narrower plot
    plt.bar(detector_data['mutation_name'], detector_data['score'])
    plt.title(f'Average Score for texts without references and with references: {detector}')
    plt.xlabel('Text type')
    plt.ylabel('Average Score')
    plt.xticks(rotation=45, ha='right')  # Rotate and align right for better readability
    plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin to prevent cutting off labels
    plt.tight_layout()
    filename = f'histogram_averages_{detector}.png'
    full_path = f'{save_folder}/{filename}'
    plt.savefig(full_path)
    plt.close()

# Now, plot the boxplots
unique_detectors = df['detector'].unique()
for detector in unique_detectors:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='mutation_name', y='score', data=df[df['detector'] == detector])
    plt.title(f'Score Distribution by text type for Detector: {detector}')
    plt.xlabel('Text type')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    filename = f'boxplot_{detector}.png'
    full_path = f'{save_folder}/{filename}'
    plt.savefig(full_path)
    plt.close()