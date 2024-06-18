from collections import defaultdict
from experiment import Experiment
from scripts.generators.parameters import Parameters
import logging
import pandas as pd
from itertools import islice
from pathlib import Path


class FormalityDetailedArgumentativeExperiment(Experiment):
    def __init__(self):
        super().__init__(name="Formality detailed argumentative test", desc="Testing different essay formality levels by specifying "
                                                              "specific characteristics for argumentative essay style.",
                         topics_split=20)
        self.formality_levels = ["informal", "semi-formal", "formal", "academic", "professional"]
        self.formality_characteristics = self.load_formalities()
        self.tokens = 800

    def generate(self):
        train_prompts = defaultdict(list)
        style = "arg"
        for formality_level in self.formality_levels:
            for topic in self.topics[style]:
                if formality_level == "Professional":
                    train_prompt = f"Write a four-hundred-word {self.topic_categories[style]} " \
                                   f"essay on the topic '{topic}' " \
                                   f"using {formality_level} tone as a Lawyer. Follow the style and implement " \
                                   f"the following rules in the essay: '{self.formality_characteristics[formality_level]}'."
                else:
                    train_prompt = f"Write a four-hundred-word {self.topic_categories[style]} " \
                                   f"essay on the topic '{topic}' " \
                                   f"using {formality_level} tone. Follow the style and implement " \
                                   f"the following rules in the essay: '{self.formality_characteristics[formality_level]}'."

                train_prompts[formality_level].append(train_prompt)

        content = {}
        parameters = Parameters(max_tokens=800)
        for formality_level, prompts in train_prompts.items():
            for index in range(len(prompts)):
                ID = f"form_test_{formality_level}_p_{index}"
                prompt = prompts[index]
                messages = [{"role": "user", "content": f"{prompt}"}]
                content[f"{ID}"] = {"messages": messages, "parameters": parameters}
        logging.info(content)
        result = self.generators[0].generate_batch(content, threads=10)
        self.save_texts(result)

    def load_formality_characteristics(self, formality_level):
        script_path = Path(__file__).resolve()
        formality_folder = script_path.parent.parent.parent / "data/formality"
        file_path = formality_folder / f"{formality_level}.txt"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.readlines()
        command = ""
        for i in range(len(content)):
            content[i] = content[i].replace('\n', '')
            command = f"{command} {content[i]}"
        return command

    def load_formalities(self):
        formalities = {}
        for formality in self.formality_levels:
            formalities[formality] = self.load_formality_characteristics(formality)
        return formalities


experiment = FormalityDetailedArgumentativeExperiment()
experiment.detect()
