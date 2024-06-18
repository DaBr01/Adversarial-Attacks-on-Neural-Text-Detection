import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

# change experiment folder name to do visuals for other formality family tests.
df = pd.read_csv("../../data/Basic Prompt engineering/results/result.csv")
save_folder = "..\..\data\Basic Prompt engineering\\results"


def get_prompt_name(text_id):
    mapping = {
            "prt_std": "Baseline prompt",
            "prt_av": "Avoid detection prompt",
            "prt_sma": "Smart prompt",
            "prt_expl": "Internet prompt",
            "prt_perpl": "Perplexity prompt",

            "prt_stdreg": "Baseline regen",
            "prt_avreg": "Regen avoid detection",
            "prt_smareg": "Regen smart prompt",
            "prt_explreg": "Regen internet prompt",
            "prt_perplreg": "Regen perplexity prompt",
        }
    codes = text_id.split("_")[0:2]
    code = "_".join(codes)
    return mapping[code]


df["Prompt Experiment"] = df["text_id"].apply(get_prompt_name)
detectors = df['detector'].unique()
df['score'] = pd.to_numeric(df['score'], errors='coerce')
print(df.head())
df.to_csv(os.path.join(save_folder, "basic_prompt_engineering.csv"))
for detector in detectors:
    subset = df[df['detector'] == detector]
    # Group by 'Prompt Experiment' and calculate mean scores
    grouped = subset.groupby('Prompt Experiment')['score'].mean().reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(grouped['Prompt Experiment'], grouped['score'], color='skyblue')
    plt.title(f"Average Detection Rates for {detector}")
    plt.xlabel('Prompt Experiment')
    plt.ylabel('Average Score')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    filename = f'histogram_averages_{detector}.png'
    full_path = f'{save_folder}/{filename}'
    plt.savefig(full_path)
    plt.close()

for detector in detectors:
    subset = df[df['detector'] == detector]

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Prompt Experiment', y='score', data=subset)
    plt.title(f"Score Distribution for {detector}")
    plt.xlabel('Prompt Experiment')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    filename = f'boxplot_{detector}.png'
    full_path = f'{save_folder}/{filename}'
    plt.savefig(full_path)
    plt.close()