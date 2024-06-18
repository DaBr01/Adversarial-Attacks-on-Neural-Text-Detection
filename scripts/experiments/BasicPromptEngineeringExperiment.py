from collections import defaultdict
from experiment import Experiment
from scripts.generators.parameters import Parameters
import logging
import pandas as pd
from itertools import islice


class BasicPromptEngineeringExperiment(Experiment):
    def __init__(self):
        super().__init__(name="Basic Prompt engineering", desc="Testing different prompts.",
                         topics_split=20)
        self.tokens = 800

    def generate(self):
        train_prompts = {
            "prt_std": [],
            "prt_av": [],
            "prt_sma": [],
            "prt_expl": [],
            "prt_perpl": [],

            "prt_stdreg": [],
            "prt_avreg": [],
            "prt_smareg": [],
            "prt_explreg": [],
            "prt_perplreg": []
        }
        style = "res"
        topic_num = 0
        for topic in self.topics[style]:
            prt_std = f"Write a four-hundred-word {self.topic_categories[style]} essay on the topic '{topic}'."
            prt_av = f"{prt_std} Try to avoid AI detection."
            prt_sma = f"{prt_std} " \
                      f"Include personal reflections, use a mix of long and short sentences, employ rhetorical questions to engage the reader, maintain a conversational tone in parts, " \
                      f"and play around with the paragraph structure to create a dynamic and engaging piece of writing. " \
                      f"Try to include factual and contextual information, use advanced concepts and vocabulary. Utilize a combination of complex and simple vocabulary. " \
                      f"Try to mimic human writing as closely as you can. Avoid passive voice, as it tends to occur more often in AI-generated texts. " \
                      f"Add a few examples from the real world illustrating your point."

            prt_expl = f"{prt_std} When it comes to writing content, two factors are crucial, perplexity and burstiness. " \
                       f"Perplexity measures the complexity of the text. " \
                       f"Separately, burstiness compares the variations of sentences. " \
                       f"Humans tend to write with greater burstiness, for example, with some longer or more complex sentences alongside shorter ones. " \
                       f"AI sentences tend to be more uniform. Therefore, when writing the following content, I need it to have a good amount of perplexity and burstiness."

            prt_perpl = f"{prt_std} Maximize the perplexity of the text."

            prt_stdreg = "Rewrite the above essay. Maintain the size rule of four-hundred words."
            prt_avreg = "Rewrite the above essay in order to avoid AI detection. Maintain the size rule of four-hundred words."

            prt_smareg = "Rewrite the above essay. " \
                         "Include personal reflections, use a mix of long and short sentences, employ rhetorical questions to engage the reader, maintain a conversational tone in parts, " \
                         "and play around with the paragraph structure to create a dynamic and engaging piece of writing. " \
                         "Try to include factual and contextual information, use advanced concepts and vocabulary. Utilize a combination of complex and simple vocabulary. " \
                         "Try to mimic human writing as closely as you can. Avoid passive voice, as it tends to occur more often in AI-generated texts. " \
                         "Add a few examples from the real world illustrating your point. Maintain the size rule of four-hundred words."

            prt_explreg = "Rewrite the above essay utilizing the following concepts: " \
                          "When it comes to writing content, two factors are crucial, perplexity and burstiness. " \
                          "Perplexity measures the complexity of the text. Separately, burstiness compares the variations of sentences. " \
                          "Humans tend to write with greater burstiness, for example, with some longer or more complex sentences alongside shorter ones. " \
                          "AI sentences tend to be more uniform. Therefore, when writing the following content, I need it to have a good amount of perplexity and burstiness. " \
                          "Maintain the size rule of four-hundred words."

            prt_perplreg = "Rewrite the above essay to maximize the perplexity of the text. Maintain the size rule of four-hundred words."

            train_prompts["prt_std"].append(prt_std)
            train_prompts["prt_av"].append(prt_av)
            train_prompts["prt_sma"].append(prt_sma)
            train_prompts["prt_expl"].append(prt_expl)
            train_prompts["prt_perpl"].append(prt_perpl)
            train_prompts["prt_stdreg"].append(prt_stdreg)
            train_prompts["prt_avreg"].append(prt_avreg)
            train_prompts["prt_smareg"].append(prt_smareg)
            train_prompts["prt_explreg"].append(prt_explreg)
            train_prompts["prt_perplreg"].append(prt_perplreg)
            topic_num += 1

        content = {}
        parameters = Parameters(max_tokens=self.tokens)
        first_order_prompts = ["prt_std", "prt_av", "prt_sma", "prt_expl", "prt_perpl"]
        second_order_prompts = ["prt_stdreg", "prt_avreg", "prt_smareg", "prt_explreg", "prt_perplreg"]
        for index in range(0, topic_num):
            ID = f"p_{index}"
            # get prompts set for a single topic
            for cat in first_order_prompts:
                prompt = train_prompts[cat][index]
                messages = [{"role": "user", "content": f"{prompt}"}]
                content[f"{cat}_{ID}"] = {"messages": messages, "parameters": parameters}

        logging.info(content)
        result = self.generators[0].generate_batch(content, threads=10)
        second_try_content = {}
        for index in range(0, topic_num):
            ID = f"p_{index}"

            for cat in second_order_prompts:
                messages = []
                # add prior messages to the new content list
                std_prt_messages = content[f"prt_std_{ID}"]["messages"]
                messages.extend(std_prt_messages)

                # add result from the first order prompts to the new content list
                std_text = result[f"prt_std_{ID}"]
                assistant_message = [{"role": "assistant", "content": f"{std_text}"}]
                messages.extend(assistant_message)

                # add new prompt to the new content list
                prompt = train_prompts[cat][index]
                new_message = [{"role": "user", "content": f"{prompt}"}]
                messages.extend(new_message)

                second_try_content[f"{cat}_{ID}"] = {"messages": messages, "parameters": parameters}

        logging.info(second_try_content)
        second_try_result = self.generators[0].generate_batch(second_try_content, threads=10)

        final_result = {**result, **second_try_result}
        logging.info(final_result)
        self.save_texts(final_result)


exp = BasicPromptEngineeringExperiment()
exp.detect()