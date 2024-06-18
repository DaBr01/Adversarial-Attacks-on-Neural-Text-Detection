import transformers
import torch
import torch.nn.functional as F
import numpy as np
import random
from scripts.detectors.detector import Detector
from concurrent.futures import ThreadPoolExecutor, as_completed


class RadarDetector(Detector):
    def __init__(self):
        super().__init__()
        self.device = "cuda"
        model_name = "TrustSafeAI/RADAR-Vicuna-7B"
        self.detector = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.detector.eval()
        self.detector.to(self.device)
        self.name = "Radar"

    # code borrowed from the official github of the paper, showcasing the demo.
    def detect(self, text):
        Text_input = [text]
        with torch.no_grad():
            inputs = self.tokenizer(Text_input, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            output_probs = F.log_softmax(self.detector(**inputs).logits, -1)[:, 0].exp().tolist()
        return output_probs[0]

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
