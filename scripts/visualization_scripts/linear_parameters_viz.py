import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# change experiment folder name to do visuals for other formality family tests.
df = pd.read_csv("..\..\data\Linear Search\\results\\result.csv")
save_folder = "..\..\data\Linear Search\\results"

df["parameter name"] = df["text_id"].apply(lambda x: x.split("_")[0])
df["parameter value"] = df["text_id"].apply(lambda x: x.split("_")[1])

print(df.head())
# Getting unique detectors
detectors = df['detector'].unique()

df['parameter value'] = pd.to_numeric(df['parameter value'], errors='coerce')
df['score'] = pd.to_numeric(df['score'], errors='coerce')
df.to_csv(f'{save_folder}/linear_search_preliminary.csv', index=False)
# Group the data by 'parameter name', 'detector', and 'parameter value' and calculate the average score
df_avg = df.groupby(['parameter name', 'detector', 'parameter value'])['score'].mean().reset_index()

# Get unique parameter names
parameters = df_avg['parameter name'].unique()

for param in parameters:
    # Filter the DataFrame for the current parameter
    df_param = df_avg[df_avg['parameter name'] == param]

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # For each detector, plot parameter value vs. average score
    for detector in df_param['detector'].unique():
        df_detector = df_param[df_param['detector'] == detector]
        # Sort by 'parameter value' for consistent plotting
        df_detector = df_detector.sort_values(by='parameter value')

        ax.plot(df_detector['parameter value'], df_detector['score'], marker='o', label=detector)

    # Setting the plot title and labels
    ax.set_title(f'Average Score vs Parameter Value for {param}')
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Average Score')

    # Adding a legend with the title 'Detector'
    ax.legend(title='Detector', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adding a grid for better readability
    ax.grid(True)

    # Use a tight layout for the plot
    plt.tight_layout()
    # Constructing the filename based on the detector name
    filename = f'line_plot_{param}.png'
    full_path = f'{save_folder}/{filename}'

    # Saving the plot to the specified location
    plt.savefig(full_path)

    # Closing the plot to free memory
    plt.close()

