import pandas as pd
import matplotlib.pyplot as plt
import os

# change experiment folder name to do visuals for other formality family tests.
df = pd.read_csv("..\..\data\Formality detailed argumentative test\\results\\result.csv")
save_folder = "..\..\data\Formality detailed argumentative test\\results"


def get_formality(id):
    id = id.split("_")[2:-2][0]
    return id


df["formality"] = df["text_id"].apply(get_formality)
print(df.head())
df.to_csv(os.path.join(save_folder, "formality_detailed_test_argumentative.csv"))
# Getting unique detectors
detectors = df['detector'].unique()

for detector in detectors:
    # Filtering the DataFrame for the current detector
    subset_df = df[df['detector'] == detector]

    # Creating a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generating a boxplot for each style within the current detector
    boxplot = subset_df.boxplot(column=['score'], by='formality', ax=ax, grid=False)

    # Setting the title and labels
    ax.set_title(f'Scores by Formality (using detailed prompt) for Detector: {detector}')
    plt.suptitle('')  # Suppress the default subtitle to avoid redundancy
    ax.set_xlabel('Formality')
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

avg_scores = df.groupby(['detector', 'formality'])['score'].mean().reset_index()

# Pivot the data so that detectors are columns, styles are rows, and values are average scores
pivot_table = avg_scores.pivot(index='formality', columns='detector', values='score')

# Plotting
pivot_table.plot(kind='bar', figsize=(12, 8))

plt.title('Average Detection Score for Each Formality (using detailed prompts) Across Detectors')
plt.xlabel('Formality')
plt.ylabel('Average Score')
plt.xticks(rotation=45)
plt.legend(title='Detector')

# Adjust layout
plt.tight_layout()

filename = f'histogram_averages.png'
full_path = f'{save_folder}/{filename}'
plt.savefig(full_path)
plt.close()
