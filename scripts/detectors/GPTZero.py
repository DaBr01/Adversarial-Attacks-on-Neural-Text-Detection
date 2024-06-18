from scripts.detectors.detector import Detector
from gptzero import GPTZeroAPI
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback


class GPTZero(Detector):
    def __init__(self):
        super().__init__()
        self.api_key = ""
        self.gptzero_api = GPTZeroAPI(api_key=self.api_key)
        self.name = "GPTZero"

    def detect(self, text):
        exceptions_num = 0
        while True:
            if exceptions_num > 5:
                print("More than 6 sequential errors, returning None")
                return None
            try:
                response = self.gptzero_api.text_predict(text)['documents'][0]['class_probabilities']['ai']
                print(f"GPTZero Prediction: {response}")
                return response
            except Exception as exc:
                exceptions_num += 1
                print(f"GPTZero stumbled upon exception, retrying... Exception: {exc}")

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

