from collections import defaultdict

from experiment import Experiment
from scripts.detectors.gpt2detector import GPT2Detector
from scripts.generators.generator import Generator
from scripts.generators.parameters import Parameters
import numpy as np
import logging
from pathlib import Path


class GridSearchFPPPOnly(Experiment):
    def __init__(self):
        super().__init__(name="Grid search fp pp only pp1-2", desc="performing grid search", topics_split=5)
        self.values = {
            'frequency_penalty': {'fixed_value': 0, 'bounds': [0, 1], 'step': 0.1, "short_name": "fp"},
            'presence_penalty': {'fixed_value': 0, 'bounds': [1.1, 2], 'step': 0.1, "short_name": "pp"}
        }
        self.tokens = 800

    def generate(self):
        train_prompts = defaultdict(list)
        styles = ["res"]
        for style in styles:
            for topic in self.topics[style]:
                train_prompt = f"Write a four-hundred-word {self.topic_categories[style]} essay on the topic '{topic}'."
                train_prompts[style].append(train_prompt)

        content = {}
        # iterate over all 4 parameters
        bounds = self.values["frequency_penalty"]["bounds"]
        bounds_pp = self.values["presence_penalty"]["bounds"]
        step = self.values["frequency_penalty"]["step"]
        par_range = [round(i, 1) for i in np.arange(bounds[0], bounds[1] + step, step)]
        pp_range = [round(i, 1) for i in np.arange(bounds_pp[0], bounds_pp[1] + step, step)]
        for fp in par_range:
            for pp in pp_range:
                parameters = Parameters(max_tokens=self.tokens, frequency_penalty=fp, presence_penalty=pp)
                for category, prompts in train_prompts.items():
                    for index in range(len(prompts)):
                        ID = f"grs_fp_{fp}_{pp}_{category}_{index}"
                        prompt = prompts[index]
                        messages = [{"role": "user", "content": f"{prompt}"}]
                        content[f"{ID}"] = {"messages": messages, "parameters": parameters}
        logging.info(content)
        result = self.generators[0].generate_batch(content, threads=10)
        self.save_texts(result)


grid_search = GridSearchFPPPOnly()
grid_search.detect()
