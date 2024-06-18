from openai import OpenAI, OpenAIError, RateLimitError
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import regex


class Generator:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def generate_text(self, messages, parameters):
        print(f"Generating text for messages: {messages}, parameters: {parameters.__dict__}")
        num_errors = 0
        while True:
            while True:
                try:
                    if num_errors > 10:
                        print("Stuck in loop")
                        return None
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=parameters.temperature,
                        top_p=parameters.top_p,
                        n=parameters.n,
                        max_tokens=parameters.max_tokens,
                        presence_penalty=parameters.presence_penalty,
                        frequency_penalty=parameters.frequency_penalty
                    )
                    break
                except RateLimitError:
                    print("RateLimitError encountered, retrying")
                    time.sleep(20)
                    num_errors += 1
                except OpenAIError as ecs:
                    print(f"OpenAIError encountered, retrying: {ecs}")
                    num_errors += 1
            reply = response.choices[0].message.content
            word_count = len(reply.split())
            if 300 <= word_count <= 500:
                break
            print(f"Number of words insufficient: {word_count}, regenerating text.")
        return reply

    # parameters: content = {"ID": {"messages":[messages], "parameters":Parameters}}
    def generate_batch(self, content, threads=1):
        results = {}
        # Create a thread pool with the specified number of threads
        with ThreadPoolExecutor(max_workers=threads) as executor:
            # Dictionary to keep track of future to key mapping
            future_to_result = {
                executor.submit(self.generate_text, values["messages"], values["parameters"]): key
                for key, values in content.items()
            }

            # As each future completes, get the result and update the results dictionary
            for future in as_completed(future_to_result):
                key = future_to_result[future]
                try:
                    result = future.result()  # Retrieves the result from the future
                    results[key] = result
                except Exception as exc:
                    print(f'{key} generated an exception: {exc}')
                    results[key] = None  # or handle the exception as needed

        return results

    def generate_batch_language(self, content, threads=1):
        results = {}
        # Create a thread pool with the specified number of threads
        with ThreadPoolExecutor(max_workers=threads) as executor:
            # Dictionary to keep track of future to key mapping
            future_to_result = {
                executor.submit(self.generate_text_language, values["messages"], values["parameters"]): key
                for key, values in content.items()
            }

            # As each future completes, get the result and update the results dictionary
            for future in as_completed(future_to_result):
                key = future_to_result[future]
                try:
                    result = future.result()  # Retrieves the result from the future
                    results[key] = result
                except Exception as exc:
                    print(f'{key} generated an exception: {exc}')
                    results[key] = None  # or handle the exception as needed

        return results

    def generate_text_language(self, messages, parameters):
        print(f"Generating text for messages: {messages}, parameters: {parameters.__dict__}")
        num_errors = 0
        while True:
            while True:
                try:
                    if num_errors > 10:
                        print("Stuck in loop")
                        return None
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=parameters.temperature,
                        top_p=parameters.top_p,
                        n=parameters.n,
                        max_tokens=parameters.max_tokens,
                        presence_penalty=parameters.presence_penalty,
                        frequency_penalty=parameters.frequency_penalty
                    )
                    break
                except RateLimitError:
                    print("RateLimitError encountered, retrying")
                    time.sleep(20)
                    num_errors += 1
                except OpenAIError as ecs:
                    print(f"OpenAIError encountered, retrying: {ecs}")
                    num_errors += 1
            reply = response.choices[0].message.content
            break
        return reply
