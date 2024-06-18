from collections import defaultdict
from experiment import Experiment
from scripts.generators.parameters import Parameters
import logging
import pandas as pd
from itertools import islice
import re
import torch
from parrot import Parrot
import nltk
import warnings
nltk.download('punkt')
warnings.filterwarnings("ignore")


class BasicParaphraseExperiment(Experiment):
    def __init__(self):
        super().__init__(name="Basic Paraphrase Experiment", desc="Testing paraphrases.",
                         topics_split=20)
        self.paraphrases = {
            "def": lambda x: x,
            "par": self.paraphrase_text,
        }
        self.tokens = 800

    def generate(self):
        train_prompts = []
        style = "res"
        for topic in self.topics[style]:
            train_prompt = f"Write a four-hundred-word {self.topic_categories[style]} " \
                           f"essay on the topic '{topic}'."
            train_prompts.append(train_prompt)

        content = {}
        parameters = Parameters(max_tokens=self.tokens)
        for index in range(0, len(train_prompts)):
            ID = f"p_{index}"
            prompt = train_prompts[index]
            messages = [{"role": "user", "content": f"{prompt}"}]
            content[f"{ID}"] = {"messages": messages, "parameters": parameters}
        logging.info(content)
        result = self.generators[0].generate_batch(content, threads=10)
        mutated_texts = {}
        for name, mutation in self.paraphrases.items():
            for key in result.keys():
                text = result[key]
                ID = f"{name}_{key}"
                mutated_texts[ID] = mutation(text)

        self.save_texts(mutated_texts)

    def capitalize_sentences(self, text):
        words = text.split()
        # Check if there are any words to capitalize
        if words:
            # Capitalize the first word
            words[0] = words[0].capitalize()
        # Join the words back into a sentence and return
        return ' '.join(words)

    def ensure_period(self, sentence):
        # Check if the sentence ends with a period
        if not sentence.endswith('.'):
            # Add a period if it doesn't end with one
            sentence += '.'
        # Return the modified or unmodified sentence
        return sentence

    def paraphrase_text(self, text):
        # Check if GPU is available and set device accordingly
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize Parrot model
        parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=device == "cuda")

        # Use nltk to split the text into sentences
        sentences = nltk.sent_tokenize(text)

        paraphrased_text = []

        # Iterate over each sentence and paraphrase
        for sentence in sentences:
            if sentence.strip():  # ensure the sentence is not just empty spaces
                # Generate paraphrases; assuming the first returned paraphrase is used
                paraphrases = parrot.augment(input_phrase=sentence.strip())
                # Choose the best paraphrase (usually the first one)
                if paraphrases:
                    best_paraphrase = paraphrases[0][0]
                    # Capitalize the first letter of each sentence in the paraphrase
                    best_paraphrase = self.capitalize_sentences(best_paraphrase)
                    best_paraphrase = self.ensure_period(best_paraphrase)
                    paraphrased_text.append(best_paraphrase)
                else:
                    # If no paraphrase is returned, use the original sentence
                    paraphrased_text.append(self.capitalize_sentences(sentence))

        print("text paraphrased")
        print(paraphrased_text)
        # Join all paraphrased sentences
        return ' '.join(paraphrased_text)


experiment = BasicParaphraseExperiment()
experiment.detect()
