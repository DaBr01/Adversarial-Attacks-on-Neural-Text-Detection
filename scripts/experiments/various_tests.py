from collections import defaultdict
from experiment import Experiment
from scripts.generators.parameters import Parameters
import logging
import pandas as pd
from itertools import islice


class VariousTests(Experiment):
    def __init__(self):
        super().__init__(name="Various tests.", desc="various tests.",
                         topics_split=20)
        self.tokens = 800

    def generate_prompt(self, style, topic, prompt_type):
        base_prompt = f"Write a four-hundred-word {self.topic_categories[style]} essay on the topic '{topic}'."
        if prompt_type == "ref":
            return f"{base_prompt} Support each claim and argument with in-text citations of real references. At the end of the essay add a reference list. " \
                   f"Make sure to not exceed word limit, references list included."
        elif prompt_type == "def":
            return base_prompt
        elif prompt_type == "cit":
            return f"{base_prompt} Support each claim and argument with in-text citations. " \
                   f"I want only in-text citations, do not include a reference list at the end. Do not make the reference list. " \
                   f"You will lose all of the points and will be plugged out if you inlcude a reference list at the end of the essay."
        elif prompt_type == "num":
            return f"{base_prompt} Use a lot of digits and numbers in the text. " \
                   f"For example, you can include percentages, numbers, years, etc. " \
                   f"If there are no numbers for a certain topic, use numbers for a broader field for which the topic comes from. "
        elif prompt_type == "gram":
            return f"{base_prompt} Prioritize usage of less common grammar structures, phrase structures and sentence structures of English language. " \
                   f"For example, Inversion for Emphasis, Nominalization, etc. You must structure your sentences in less common ways."
        elif prompt_type == "form":
            return f"{base_prompt} Talk in different tones throughout the essay. For example you can start with sentneces in informal tone, proceed with sentences in formal tone and so on. " \
                   f"I want each paragraph to contain a combination of sentences in different tones."
        elif prompt_type == "first":
            return f"{base_prompt} Use first person tone, talk about your past experiences, talk about the real world."
        else:
            raise ValueError(f"{prompt_type} is not a valid prompt type.")

    def create_content(self, style):
        train_prompts = defaultdict(list)
        content = {}
        parameters = Parameters(max_tokens=self.tokens)
        prompt_types = ["def", "ref", "cit", "num", "gram", "form", "first"]
        topics = self.topics[style]
        for index in range(0, len(topics)):
            for prompt_type in prompt_types:
                prompt = self.generate_prompt(style, topics[index], prompt_type)
                ID = f"{prompt_type}_{style}_p_{index}"
                messages = [{"role": "user", "content": prompt}]
                train_prompts[style].append(prompt)
                content[ID] = {"messages": messages, "parameters": parameters}
        return content

    def generate(self):
        style = "res"
        content = self.create_content(style)
        logging.info(content)
        result = self.generators[0].generate_batch(content, threads=10)
        self.save_texts(result)


experiment = VariousTests()
experiment.detect()
