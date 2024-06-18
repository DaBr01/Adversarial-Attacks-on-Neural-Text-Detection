from scripts.detectors.gpt2detector import GPT2Detector
from scripts.detectors.GPTZero import GPTZero
from scripts.detectors.CopyLeaks import CopyLeaks
from scripts.detectors.GPT2DetectorModel import GPT2Detector
from scripts.detectors.radar import RadarDetector
from scripts.generators.generator import Generator
from scripts.generators.parameters import Parameters
import numpy as np
import pandas as pd
import os
import logging
from pathlib import Path


class Experiment:
    def __init__(self, name, desc, topics_split):
        self.name = name
        self.desc = desc
        self.topic_categories = {
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
        self.script_path = Path(__file__).resolve()
        self.data_folder = self.script_path.parent.parent.parent/f"data/{name}"
        self.results_folder = self.data_folder / "results"
        self.texts_folder = self.data_folder / "texts"
        self.prompts_folder = self.data_folder / "prompts"
        self.topics = self.load_topics(topics_split)
        self.generators = [
            Generator(
                api_key="",
                model="gpt-3.5-turbo-0125"
            )
        ]
        self.detectors = {CopyLeaks(), GPT2Detector(), GPTZero(), RadarDetector()}
        # create a folder for experiment results, if doesn't exist
        if not self.data_folder.exists():
            self.data_folder.mkdir(parents=True)
            self.results_folder.mkdir(parents=True)
            self.texts_folder.mkdir(parents=True)
            self.prompts_folder.mkdir(parents=True)

        # logger
        logging.basicConfig(filename='experiment.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.DEBUG)

    def load_topics(self, topics_split):
        topics = {}
        for cat_short, cat_normal in self.topic_categories.items():
            file_path = self.data_folder.parent / f"topics/{cat_normal}.txt"
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.readlines()

            for i in range(len(content)):
                content[i] = content[i].replace('\n', '')
            topics[cat_short] = content[0:topics_split]
        return topics

    def generate(self):
        file_names = []

    def save_results(self, df):
        df.to_csv(os.path.join(self.results_folder, "result.csv"), index=False)

    def detect(self):
        texts = self.load_texts()
        results = []
        for detector in self.detectors:
            output = detector.detect_batch(texts, threads=10)
            detector_results = [(detector.name, text_id, score) for text_id, score in output.items()]
            results.extend(detector_results)
        df = pd.DataFrame(results, columns=['detector', 'text_id', 'score'])
        self.save_results(df)

    def load_texts(self):
        texts = {}
        # Loop through each file in the specified directory
        for filename in os.listdir(self.texts_folder):
            # Check if the file is a .txt file
            if filename.endswith('.txt'):
                # Construct the full path to the file
                filepath = os.path.join(self.texts_folder, filename)
                # Open and read the content of the file
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # Use the filename without the extension as the key and the file content as the value
                    texts[os.path.splitext(filename)[0]] = content
        return texts

    def save_texts(self, content):
        for key, val in content.items():
            file_path = self.texts_folder
            file_path.mkdir(parents=True, exist_ok=True)
            file_path = file_path / f"{key}.txt"
            with open(file_path, "w", encoding='utf-8') as file:
                file.write(val)

