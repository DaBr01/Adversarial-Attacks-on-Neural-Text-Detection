import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import seaborn as sns

# change experiment folder name to do visuals for other formality family tests.
df = pd.read_csv("../../data/Grid search fp pp only pp1-2/results/result.csv")
save_folder = "..\..\data\Grid search fp pp only pp1-2\\results"

df["fp"] = df["text_id"].apply(lambda x: x.split("_")[2])
df["pp"] = df["text_id"].apply(lambda x: x.split("_")[3])

print(df.head())
# Getting unique detectors
detectors = df['detector'].unique()

df['fp'] = pd.to_numeric(df['fp'], errors='coerce')
df['pp'] = pd.to_numeric(df['pp'], errors='coerce')
df['score'] = pd.to_numeric(df['score'], errors='coerce')

print(df.head())
df.to_csv(os.path.join(save_folder, "grid_search_fp_pp_only.csv"))
# check if empty scores
empty_score_text_ids = df[pd.isna(df['score'])]['text_id']

# Setting up the color map
cmap = matplotlib.colormaps['viridis']

# Iterating through each detector
for detector in detectors:
    # Filter DataFrame
    df_filtered = df[df['detector'] == detector]
    mean_scores = df_filtered.groupby(['fp', 'pp'])['score'].mean().reset_index()
    norm_scores = (mean_scores['score'] - mean_scores['score'].min()) / (
                mean_scores['score'].max() - mean_scores['score'].min())

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(mean_scores['pp'], mean_scores['fp'], c=norm_scores, cmap=cmap, edgecolor='black', s=200)

    # Adding colorbar
    fig.colorbar(scatter, ax=ax, label='Mean Score')

    ax.set_title(f'Detector: {detector}')
    ax.set_xlabel('Presence Penalty (pp)')
    ax.set_ylabel('Frequency Penalty (fp)')
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(True)

    # Constructing the filename based on the detector name
    filename = f'heatmap_{detector.lower()}.png'
    full_path = f'{save_folder}/{filename}'

    # Saving the plot to the specified location
    plt.savefig(full_path)

    # Closing the plot to free memory
    plt.close()