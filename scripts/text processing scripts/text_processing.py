import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scripts.experiments.experiment import Experiment
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

experiment_names = ["Style Difference Test", "Formality detailed test",
                    "Formality detailed argumentative test", "Formality test"]
save_folder = f"..\..\scripts\\text processing scripts\\texts"

final_df = pd.DataFrame()
for experiment_name in experiment_names:
    df = pd.read_csv(f"..\..\data\\{experiment_name}\\results\\result.csv")
    df = df[df["detector"] == "GPTZero"].reset_index(drop=True)
    exp = Experiment(name=experiment_name, desc="", topics_split=1)
    texts = exp.load_texts()
    texts_df = pd.DataFrame(list(texts.items()), columns=['text_id', 'text'])
    merged_df = pd.merge(df, texts_df, on='text_id', how='left')
    merged_df["word count"] = merged_df["text"].apply(lambda x: len(x.split()))

    # Add the experiment_name column
    merged_df["experiment_name"] = experiment_name

    # Append to the final DataFrame
    final_df = pd.concat([final_df, merged_df], ignore_index=True)


def calculate_text_metrics(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    # Average Word Length
    average_word_length = sum(len(word) for word in words) / len(words)

    # Lexical Diversity
    lexical_diversity = len(set(words)) / len(words) if words else 0

    return pd.Series({
        'average_word_length': average_word_length,
        'lexical_diversity': lexical_diversity,
    })


def sentence_complexity_balance(text):
    sentences = sent_tokenize(text)
    sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]

    # Basic complexity measure: average sentence length
    average_length = sum(sentence_lengths) / len(sentence_lengths)

    # Variability in sentence length
    length_variability = statistics.stdev(sentence_lengths)

    return pd.Series({"average_sentence_length": average_length, "sentence_length_variability": length_variability})

final_df[["average_sentence_length", "sentence_length_variability"]] = final_df["text"].apply(sentence_complexity_balance)
final_df[["average_word_length", "lexical_diversity"]] = final_df["text"].apply(calculate_text_metrics)

final_df.to_csv(f"{save_folder}/merged_result.csv", index=False)

# List of variables to check correlation with 'score'
variables = ['word count', 'average_sentence_length', 'sentence_length_variability', 'average_word_length',
             'lexical_diversity']

# Initialize a dictionary to hold the correlation values
correlations = {}
df = final_df
for variable in variables:
    spearman_corr = df['score'].corr(df[variable], method='spearman')

    # Store the correlation values in the dictionary
    correlations[variable] = {'Spearman': spearman_corr}

# Convert the dictionary to a DataFrame for better visualization
correlations_df = pd.DataFrame(correlations).T  # Transpose for better readability

low_scores = final_df[final_df["score"] < 0.5].copy()
high_scores = final_df[final_df["score"] >= 0.9].copy()

# Add a new column to indicate the score group
low_scores.loc[:, 'Score Group'] = 'Low (<0.5)'
high_scores.loc[:, 'Score Group'] = 'High (>=0.9)'
print(low_scores.head(20))
for index, row in low_scores.iterrows():
    # Define the filename using text_id and append .txt
    file_path = f"{save_folder}/{row['text_id']}.txt"

    # Write the text to a text file
    with open(file_path, 'w') as file:
        file.write(row['text'])
# Combine the two DataFrames back for easier plotting
combined_df = pd.concat([low_scores, high_scores], axis=0)

# Selecting the variables of interest and melting the DataFrame
melted_df = combined_df.melt(id_vars=['Score Group'],
                             value_vars=['word count', 'average_sentence_length',
                                         'sentence_length_variability', 'average_word_length',
                                         'lexical_diversity'],
                             var_name='Variable', value_name='Value')

variables = ['word count', 'average_sentence_length', 'sentence_length_variability', 'average_word_length',
             'lexical_diversity']

# Iterate over each variable to create a separate box plot
for variable in variables:
    # Filter melted_df for the current variable
    filtered_df = melted_df[melted_df['Variable'] == variable]

    # Create the box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Score Group', y='Value', data=filtered_df)
    plt.title(f'Box Plot of {variable}')
    plt.xlabel('Score Group')
    plt.ylabel(variable)
    plt.show()