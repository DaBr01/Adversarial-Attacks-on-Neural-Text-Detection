import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("..\..\data\Style Difference Test\\results\\result.csv")
save_folder = "..\..\data\Style Difference Test\\results"
topic_categories = {
        'arg': 'argumentative',
        'cause_eff': 'cause_and_effect',
        'comp_contr': 'compare_contrast',
        'contr_arg': 'controversial_argumentative',
        'desc': 'descriptive',
        'expos': 'expository',
        'funny_arg': 'funny_argumentative',
        'nar': 'narrative',
        'pers': 'persuasive',
        'res': 'research'
    }


def get_category(id):
    id = id.split("_")[3:-2]
    if len(id) > 1:
        cat_id = f"{id[0]}_{id[1]}"
    else:
        cat_id = f"{id[0]}"
    return cat_id


df["style"] = df["text_id"].apply(get_category)
df.to_csv(os.path.join(save_folder, "style_difference_test.csv"))
# Getting unique detectors
detectors = df['detector'].unique()

for detector in detectors:
    # Filtering the DataFrame for the current detector
    subset_df = df[df['detector'] == detector]

    # Creating a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generating a boxplot for each style within the current detector
    boxplot = subset_df.boxplot(column=['score'], by='style', ax=ax, grid=False)

    # Setting the title and labels
    ax.set_title(f'Scores by Style for Detector: {detector}')
    plt.suptitle('')  # Suppress the default subtitle to avoid redundancy
    ax.set_xlabel('Style')
    ax.set_ylabel('Score')

    # Rotating x-axis labels and adjusting layout
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()

    # Constructing the filename based on the detector name
    filename = f'box_plot_{detector.lower()}.png'
    full_path = f'{save_folder}/{filename}'

    # Saving the plot to the specified location
    plt.savefig(full_path)

    # Closing the plot to free memory
    plt.close()

avg_scores = df.groupby(['detector', 'style'])['score'].mean().reset_index()

# Pivot the data so that detectors are columns, styles are rows, and values are average scores
pivot_table = avg_scores.pivot(index='style', columns='detector', values='score')

# Plotting
pivot_table.plot(kind='bar', figsize=(12, 8))

plt.title('Average Detection Score for Each Style Across Detectors')
plt.xlabel('Style')
plt.ylabel('Average Score')
plt.xticks(rotation=45)
plt.legend(title='Detector')

# Adjust layout
plt.tight_layout()

filename = f'histogram_averages.png'
full_path = f'{save_folder}/{filename}'
plt.savefig(full_path)
plt.close()
