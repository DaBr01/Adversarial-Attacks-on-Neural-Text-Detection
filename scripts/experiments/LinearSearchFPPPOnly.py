from collections import defaultdict

from experiment import Experiment
from scripts.detectors.gpt2detector import GPT2Detector
from scripts.generators.generator import Generator
from scripts.generators.parameters import Parameters
import numpy as np
import logging
from pathlib import Path


class LinearSearchFPPPOnly(Experiment):
    def __init__(self):
        super().__init__(name="Linear Search fp pp only", desc="performing linear search for 2 categories.", topics_split=5)
        self.values = {
            'frequency_penalty': {'fixed_value': 0, 'bounds': [0, 2], 'step': 0.1, "short_name": "fp"},
            'presence_penalty': {'fixed_value': 0, 'bounds': [0, 2], 'step': 0.1, "short_name": "pp"}
        }
        self.tokens = 800

    def generate(self):
        train_prompts = defaultdict(list)
        styles = ["arg", "res"]
        for style in styles:
            for topic in self.topics[style]:
                train_prompt = f"Write a four-hundred-word {self.topic_categories[style]} essay on the topic '{topic}'."
                train_prompts[style].append(train_prompt)

        content = {}
        # iterate over all 4 parameters
        for parameter_name in self.values.keys():
            id_name = self.values[parameter_name]["short_name"]
            # get all values for the parameter to iterate over
            lower_bound = self.values[parameter_name]["bounds"][0]
            upper_bound = self.values[parameter_name]["bounds"][1]
            step = self.values[parameter_name]["step"]
            param_values = np.arange(lower_bound, upper_bound + step, step)
            # iterate over all values of each parameter
            for parameter_value in param_values:
                parameter_value = round(parameter_value, 3)
                # create parameter object
                parameters = Parameters(max_tokens=800)
                setattr(parameters, parameter_name, parameter_value)
                # iterate over each prompt
                for category, prompts in train_prompts.items():
                    for index in range(len(prompts)):
                        ID = f"{id_name}_{parameter_value}_{category}_{index}"
                        prompt = prompts[index]
                        messages = [{"role": "user", "content": f"{prompt}"}]
                        content[f"{ID}"] = {"messages": messages, "parameters": parameters}
        logging.info(content)
        result = self.generators[0].generate_batch(content, threads=10)
        self.save_texts(result)


experiment = LinearSearchFPPPOnly()
experiment.detect()
