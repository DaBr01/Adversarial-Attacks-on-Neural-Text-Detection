from collections import defaultdict
from experiment import Experiment
from scripts.generators.parameters import Parameters
import logging
import pandas as pd
from itertools import islice


class CharMutationExperiment(Experiment):
    def __init__(self):
        super().__init__(name="Char Mutation Experiment", desc="Testing different character level mutations.",
                         topics_split=20)
        self.char_mutations = {
            "def": lambda x: x,
            "l-i": self.swap_i_l,
            "cyr": self.replace_english_with_cyrilic,
            "arm": self.replace_armenian,
            "gre": self.replace_greek,
            "punc": self.replace_punctuation,
            "inv": self.replace_invisible,
            "var": self.various_replacements,
            "maj": self.major_coverage,
            "comb": self.combined_attack,
            "cyrsm": self.replace_english_with_cyrillic_simple
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
        for name, mutation in self.char_mutations.items():
            for key in result.keys():
                text = result[key]
                ID = f"{name}_{key}"
                mutated_texts[ID] = mutation(text)

        self.save_texts(mutated_texts)

    def replace_english_with_cyrilic(self, text):
        # Define the mapping of Latin letters to their Cyrillic counterparts using Unicode
        mapping = {
            'a': '\u0430',  # Cyrillic 'а'
            'c': '\u0441',  # Cyrillic 'с'
            'e': '\u0435',  # Cyrillic 'е'
            'i': '\u0456',  # Cyrillic 'і'
            'l': '\u04CF',  # Cyrillic 'ӏ' (palochka)
            'o': '\u043E',  # Cyrillic 'о'
            'p': '\u0440',  # Cyrillic 'р'
            'x': '\u0445',  # Cyrillic 'х'
            'y': '\u0443',  # Cyrillic 'у'
            'h': '\u04BB',  # Cyrillic 'һ'
            'w': '\u051D',  # Cyrillic 'ԝ'
            'j': '\u0458',  # Cyrillic 'ј'
            's': "\u0455",  # Cyrillic 'ѕ'
        }

        # Replace each Latin character with its Cyrillic counterpart
        translated_text = ''.join(mapping.get(char, char) for char in text)

        return translated_text

    def replace_english_with_cyrillic_simple(self, text):
        # Define the mapping of Latin letters to their Cyrillic counterparts using Unicode
        mapping = {
            'a': '\u0430',  # Cyrillic 'а'
            'c': '\u0441',  # Cyrillic 'с'
            'e': '\u0435',  # Cyrillic 'е'
            'i': '\u0456',  # Cyrillic 'і'
            'o': '\u043E',  # Cyrillic 'о'
            'p': '\u0440',  # Cyrillic 'р'
            'x': '\u0445',  # Cyrillic 'х'
            'y': '\u0443',  # Cyrillic 'у'
            'j': '\u0458',  # Cyrillic 'ј'
            's': "\u0455",  # Cyrillic 'ѕ'
        }

        # Replace each Latin character with its Cyrillic counterpart
        translated_text = ''.join(mapping.get(char, char) for char in text)

        return translated_text

    def replace_armenian(self, text):
        mapping = {
            "o": '\u0585', # Armenian Small Letter Oh
            "h": "\u0570", # Armenian Small Letter Ho
            "g": "\u0581", # Armenian small letter co
            "u": "\u057D", # Armenian small letter seh
            "n": "\u0578", # Armenian small letter vo
        }
        translated_text = ''.join(mapping.get(char, char) for char in text)
        return translated_text

    def replace_greek(self, text):
        # Define the mapping of English characters to Greek letters using Unicode
        greek_mapping = {
            'a': '\u03B1',  # Greek Small Letter Alpha
            'y': '\u03B3',  # Greek Small Letter Gamma
            'v': '\u03BD',  # Greek Small Letter Nu
            'o': '\u03BF',  # Greek Small Letter Omicron
            'p': '\u03C1',  # Greek Small Letter Rho
        }

        # Replace each English character with its Greek counterpart
        translated_text = ''.join(greek_mapping.get(char, char) for char in text)
        return translated_text

    def replace_punctuation(self, x):
        # Define the replacement characters
        comma_replacement = '\u201A'  # Single Low-9 Quotation Mark
        dot_replacement = '\uFF0E'  # Fullwidth Full Stop

        # Replace commas and dots in the text
        replaced_text = x.replace(',', comma_replacement).replace('.', dot_replacement)

        return replaced_text

    def replace_invisible(self, x):
        # Define invisible characters
        zero_width_space = '\u200B'  # Zero Width Space
        zero_width_non_joiner = '\u200C'  # Zero Width Non-Joiner
        zero_width_joiner = '\u200D'  # Zero Width Joiner

        # Create a string that combines all three invisible characters
        invisible_sequence = zero_width_space + zero_width_non_joiner + zero_width_joiner

        # Insert the sequence of invisible characters between each character in the input string
        modified_text = invisible_sequence.join(x)

        # Ensure the sequence is also added at the end of the text if required
        modified_text += invisible_sequence

        return modified_text

    def various_replacements(self, text):
        extra_mapping = {
            "u": "\u1D1C", # unicode ᴜ
            "d": "\u217E", # unicode ⅾ
            "g": "\u0261", # unicode ɡ
            "q": "\u051B", # unicode ԛ
            "v": "\u1D20" # unicode ᴠ
        }

        translated_text = ''.join(extra_mapping.get(char, char) for char in text)
        return translated_text

    # replace cyrilic, armenian, greek, other symbols
    def major_coverage(self, text):
        mapping = {
            'a': '\u0430',  # Cyrillic 'а'
            'c': '\u0441',  # Cyrillic 'с'
            'e': '\u0435',  # Cyrillic 'е'
            'i': '\u0456',  # Cyrillic 'і'
            'l': '\u04CF',  # Cyrillic 'ӏ' (palochka)
            'o': '\u043E',  # Cyrillic 'о'
            'p': '\u0440',  # Cyrillic 'р'
            'x': '\u0445',  # Cyrillic 'х'
            'y': '\u0443',  # Cyrillic 'у'
            'h': '\u04BB',  # Cyrillic 'һ'
            'w': '\u051D',  # Cyrillic 'ԝ'
            'j': '\u0458',  # Cyrillic 'ј'
            's': "\u0455",  # Cyrillic 'ѕ'
            "g": "\u0581",  # Armenian small letter co
            "u": "\u057D",  # Armenian small letter seh
            "n": "\u0578",  # Armenian small letter vo
            'v': '\u03BD',  # Greek Small Letter Nu
            "d": "\u217E",  # unicode ⅾ
            "q": "\u051B",  # unicode ԛ
        }
        translated_text = ''.join(mapping.get(char, char) for char in text)
        return translated_text

    # replace major coverage combined with invisible characters and punctuation replacements
    def combined_attack(self, text):
        text = self.major_coverage(text)
        text = self.replace_punctuation(text)
        text = self.replace_invisible(text)
        return text

    def swap_i_l(self, text):
        new_text = text.replace("l", "I")
        return new_text


experiment = CharMutationExperiment()
experiment.detect()
