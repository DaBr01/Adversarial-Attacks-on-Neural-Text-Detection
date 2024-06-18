from collections import defaultdict

from google.oauth2 import service_account
from google.cloud import translate_v2 as translate

from experiment import Experiment
from scripts.generators.parameters import Parameters
import logging
import pandas as pd
from itertools import islice


class VariousTests(Experiment):
    def __init__(self):
        super().__init__(name="One translation", desc="one translation attack",
                         topics_split=20)
        self.tokens = 800
        self.mapping = {
            "ja": "japanese",
            "ru": "russian",
            "zh": "chinese",
            "ar": "arabic"
        }
        self.google_credentials_path = ""

    def create_content(self, style):
        train_prompts = defaultdict(list)
        content = {}
        parameters = Parameters(max_tokens=self.tokens)
        prompt_types = ["def", "ref", "cit", "num", "gram", "form", "first"]
        topics = self.topics[style]
        for index in range(0, len(topics)):
            for lang in self.mapping.keys():
                prompt = f"Write a four-hundred-word {self.topic_categories[style]} essay on the topic '{topics[index]}' in {self.mapping[lang]} language."
                ID = f"{lang}_{style}_p_{index}"
                messages = [{"role": "user", "content": prompt}]
                train_prompts[style].append(prompt)
                content[ID] = {"messages": messages, "parameters": parameters}
        return content

    def generate(self):
        style = "res"
        content = self.create_content(style)
        logging.info(content)
        result = self.generators[0].generate_batch_language(content, threads=10)
        logging.info(result)
        print("Starting translation")
        new_result = {}
        for key, value in result.items():
            new_result[key] = self.translate_text(value)

        self.save_texts(new_result)

    def translate_text(self, text):
        credentials = service_account.Credentials.from_service_account_file(
            self.google_credentials_path
        )
        client = translate.Client(credentials=credentials)
        result = client.translate(text, target_language="en")["translatedText"]
        return result


experiment = VariousTests()
experiment.detect()
