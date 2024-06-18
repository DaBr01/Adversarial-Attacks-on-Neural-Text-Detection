from scripts.detectors.detector import Detector
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
import requests
import json


class GPT2Detector(Detector):
    def __init__(self):
        super().__init__()
        self.url = 'https://openai-openai-detector.hf.space/'
        self.name = "GPT2Detector"

    def detect(self, text):
        exceptions_num = 0
        while True:
            if exceptions_num > 5:
                print("More than 6 sequential errors, returning None")
                return None
            try:
                result_object = self.send_request(text)
                result = result_object.fake_probability
                print(f"GPT2Prediction: {result}")
                return result
            except Exception as exc:
                exceptions_num += 1
                print(f"GPT2 stumbled upon exception, retrying... Exception: {exc}")

    def detect_batch(self, content, threads=1):
        results = {}
        with ThreadPoolExecutor(max_workers=threads) as executor:
            # Dictionary to keep track of future to key mapping
            future_to_key = {executor.submit(self.detect, text): key for key, text in content.items()}

            # As each future completes, get the result and update the results dictionary
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    score = future.result()
                    results[key] = score
                except Exception as exc:
                    print(f'{key} generated an exception: {exc}')
                    results[key] = None  # or any default value you prefer
        return results

    def send_request(self, text):
        quoted_text = quote(text)
        full_url = self.url + '?' + quoted_text
        try:
            response = requests.get(full_url)
            response.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            raise Exception("Http Error:", errh)
        except requests.exceptions.ConnectionError as errc:
            raise Exception("Error Connecting:", errc)
        except requests.exceptions.Timeout as errt:
            raise Exception("Timeout Error:", errt)
        except requests.exceptions.RequestException as err:
            raise Exception("Something went wrong: ", err)

        try:
            data = json.loads(response.text)
            return GPT2Output(data['all_tokens'], data['used_tokens'],
                              data['real_probability'], data['fake_probability'])
        except json.JSONDecodeError as e:
            raise Exception('Failed to decode JSON: ', e)
        except KeyError as e:
            raise Exception('JSON did not have expected format: ', e)


class GPT2Output:
    def __init__(self, all_tokens, used_tokens, real_probability, fake_probability):
        self.all_tokens = all_tokens
        self.used_tokens = used_tokens
        self.real_probability = real_probability
        self.fake_probability = fake_probability
