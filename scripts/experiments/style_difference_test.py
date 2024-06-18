from collections import defaultdict
from experiment import Experiment
from scripts.generators.parameters import Parameters
import logging
import pandas as pd
from itertools import islice


class StyleDifferenceTest(Experiment):
    def __init__(self):
        super().__init__(name="Style Difference Test", desc="Testing different essay styles and their detection.",
                         topics_split=20)
        self.tokens = 800

    def generate(self):
        print(self.topic_categories)
        train_prompts = defaultdict(list)
        for key, values in self.topics.items():
            for topic in values:
                train_prompt = f"Write a four-hundred-word {self.topic_categories[key].replace('_', ' ')} essay on the topic '{topic}'"
                train_prompts[key].append(train_prompt)
        content = {}
        parameters = Parameters(max_tokens=800)
        for category, prompts in train_prompts.items():
            for index in range(len(prompts)):
                ID = f"style_dif_cat_{category}_p_{index}"
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


test = StyleDifferenceTest()
test.detect()
