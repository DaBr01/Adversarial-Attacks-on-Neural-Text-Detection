from collections import defaultdict
from experiment import Experiment
from scripts.generators.parameters import Parameters
import logging
import pandas as pd
from itertools import islice


class FormalityExperiment(Experiment):
    def __init__(self):
        super().__init__(name="Formality test", desc="Testing different essay formality levels.",
                         topics_split=20)
        self.formality_levels = ["informal", "semi-formal", "formal", "academic", "professional"]
        self.tokens = 800

    def generate(self):
        train_prompts = defaultdict(list)
        style = "res"
        for formality_level in self.formality_levels:
            for topic in self.topics[style]:
                if formality_level == "Professional":
                    train_prompt = f"Write a four-hundred-word {self.topic_categories[style]} " \
                                   f"essay on the topic '{topic}' " \
                                   f"using {formality_level} tone as a Lawyer."
                else:
                    train_prompt = f"Write a four-hundred-word {self.topic_categories[style]} " \
                                   f"essay on the topic '{topic}' " \
                                   f"using {formality_level} tone."
                print(train_prompt)
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

    def save_texts(self, content):
        for key, val in content.items():
            file_path = self.texts_folder
            file_path.mkdir(parents=True, exist_ok=True)
            file_path = file_path / f"{key}.txt"
            with open(file_path, "w", encoding='utf-8') as file:
                file.write(val)


experiment = FormalityExperiment()
experiment.detect()
