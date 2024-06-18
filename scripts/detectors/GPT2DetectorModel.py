from transformers import pipeline, AutoTokenizer
from scripts.detectors.detector import Detector
from concurrent.futures import ThreadPoolExecutor, as_completed


class GPT2Detector(Detector):
    def __init__(self):
        super().__init__()
        self.name = "GPT2Detector"
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")
        self.pipe = pipeline("text-classification", model="openai-community/roberta-base-openai-detector")

    def detect(self, text):
        # Tokenize the text, ensuring it doesn't exceed the max length
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

        # Convert tokens back to text string, ignoring special tokens like <s> and </s>
        truncated_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)

        # Perform classification
        result = self.pipe(truncated_text)
        print(result)

        if result[0]['label'] == "Real":
            score_ai = 1 - result[0]['score']
        else:
            score_ai = result[0]['score']
        return score_ai

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
