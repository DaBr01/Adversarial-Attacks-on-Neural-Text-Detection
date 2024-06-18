import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

# change experiment folder name to do visuals for other formality family tests.
df = pd.read_csv("../../data/Higher Level Mutations/results/result.csv")
save_folder = "..\..\data\Higher Level Mutations\\results"


def extract_mutation(text_id):
    ids = text_id.split("_")
    if len(ids) == 4:
        return "_".join(ids[0:2])
    else:
        return ids[0]


df["mutation"] = df["text_id"].apply(extract_mutation)

char_mut_names = {
    "def": "Baseline",
    "tr_chi": "translation chinese",
    "tr_jap": "translation japanese",
    "tr_rus": "translation russian",
    "tr_ar": "translation arabic"
    }
df = df[df['mutation'] != 'nltk']
print(df.head())
# Getting unique detectors
detectors = df['detector'].unique()

df['score'] = pd.to_numeric(df['score'], errors='coerce')

# check if empty scores
empty_score_text_ids = df[pd.isna(df['score'])]['text_id']
print(empty_score_text_ids)

df['mutation_name'] = df['mutation'].map(char_mut_names)
df.to_csv(os.path.join(save_folder, "translations.csv"))
# Step 3: Group by 'detector' and 'mutation_name', then calculate average scores
avg_scores = df.groupby(['detector', 'mutation_name'])['score'].mean().reset_index()

# Now, plot the histograms
unique_detectors = avg_scores['detector'].unique()
for detector in unique_detectors:
    detector_data = avg_scores[avg_scores['detector'] == detector]
    plt.figure(figsize=(10, 6))  # Adjusted for a taller and narrower plot
    plt.bar(detector_data['mutation_name'], detector_data['score'])
    plt.title(f'Average Score by Mutation for Detector: {detector}')
    plt.xlabel('Mutation Name')
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
    plt.title(f'Score Distribution by Mutation for Detector: {detector}')
    plt.xlabel('Mutation Name')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    filename = f'boxplot_{detector}.png'
    full_path = f'{save_folder}/{filename}'
    plt.savefig(full_path)
    plt.close()