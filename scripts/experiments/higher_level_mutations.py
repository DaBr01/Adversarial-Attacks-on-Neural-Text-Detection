from collections import defaultdict
from experiment import Experiment
from scripts.generators.parameters import Parameters
import logging
import pandas as pd
from itertools import islice
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import nltk
from nltk.corpus import wordnet as wn
from transformers import MarianMTModel, MarianTokenizer
from google.oauth2 import service_account
from google.cloud import translate_v2 as translate
nltk.download('wordnet')
nltk.download('omw-1.4')


###### NOTE: TEXT SIZE HERE IS DECREASED DUE TO TRANSFORMER MODELS
class HigherLevelMutations(Experiment):
    def __init__(self):
        super().__init__(name="Higher Level Mutations", desc="Testing sentence and word level mutations.",
                         topics_split=20)
        self.char_mutations = {
            "def": lambda x: x,
            "nltk": self.replace_with_synonyms_nltk,
            "tr_chi": self.translation_chinese,
            "tr_jap": self.translation_japanese,
            "tr_rus": self.translation_russian,
            "tr_ar": self.translation_arabic,
        }
        self.tokens = 800
        self.google_credentials_path = ""

    def generate(self):
        train_prompts = []
        style = "res"
        for topic in self.topics[style]:
            train_prompt = f"Write a four-hundred word {self.topic_categories[style]} " \
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
        for name, mutation in self.char_mutations.items():
            for key in result.keys():
                text = result[key]
                ID = f"{name}_{key}"
                mutated_texts[ID] = mutation(text)
        self.save_texts(mutated_texts)

    def get_wordnet_pos(self, treebank_tag):
        """Converts treebank tags to WordNet tags."""
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return None

    def replace_with_synonyms_nltk(self, text):
        # Tokenize and tag the parts of speech
        tokens = nltk.word_tokenize(text)
        tagged_tokens = nltk.pos_tag(tokens)

        replaced_words = []
        for word, tag in tagged_tokens:
            wn_tag = self.get_wordnet_pos(tag)
            if wn_tag:
                # Get the first synonym of the word if it exists
                synsets = wn.synsets(word, pos=wn_tag)
                if synsets:
                    # We take the first lemma of the first synset as a synonym
                    synonym = synsets[0].lemmas()[0].name()
                    if synonym != word:
                        replaced_words.append(synonym)
                        continue
            # If no synonym is found or part of speech is not supported, keep the original word
            replaced_words.append(word)

        return ' '.join(replaced_words)

    def translation_russian(self, text):
        return self.translate_text(text, "ru")

    def translation_chinese(self, text):
        return self.translate_text(text, "zh")

    def translation_japanese(self, text):
        return self.translate_text(text, "ja")

    def translation_arabic(self, text):
        return self.translate_text(text, "ar")

    def translate_text(self, text, intermediary_language):
        credentials = service_account.Credentials.from_service_account_file(
            self.google_credentials_path
        )
        client = translate.Client(credentials=credentials)
        interm = client.translate(text, target_language=intermediary_language)["translatedText"]
        result = client.translate(interm, target_language="en")["translatedText"]
        return result


experiment = HigherLevelMutations()
experiment.detect()
