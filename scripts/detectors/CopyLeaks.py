from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from scripts.detectors.detector import Detector


class CopyLeaks(Detector):
    def __init__(self):
        super().__init__()
        self.email = ""
        self.api_key = ""
        # change whether to use existing token, or generate a new one. Need a new one each 48 hours.
        self.auth_token = self.use_existing_token()
        self.name = "CopyLeaks"

    def generate_token(self):
        url = "https://id.copyleaks.com/v3/account/login/api"
        payload = {
            "key": self.api_key,
            "email": self.email
        }
        headers = {
            'Content-Type': "application/json",
            'Accept': "application/json"
        }
        response = requests.post(url, json=payload, headers=headers)
        print(response)
        data = response.json()
        auth_token = data.get("access_token", "")
        print(auth_token)
        return auth_token

    def use_existing_token(self):
        return ""

    def detect(self, text):
        pass

    def detect_with_id(self, text, scan_id):
        url = f"https://api.copyleaks.com/v2/writer-detector/{scan_id}/check"
        payload = {
            "text": text,
            "language": "en"
        }
        headers = {
            'Authorization': f"Bearer {self.auth_token}",
            'Content-Type': "application/json",
            'Accept': "application/json"
        }
        exceptions_num = 0
        while True:
            if exceptions_num > 5:
                print("More than 6 sequential errors, returning None")
                return None
            try:
                response = requests.post(url, json=payload, headers=headers).json()["summary"]["ai"]
                print(f"CopyLeaks Prediction: {response}")
                return response
            except Exception as exc:
                exceptions_num += 1
                print(f"Copyleaks stumbled upon exception, retrying... Exception: {exc}")

    def detect_batch(self, content, threads=1):
        results = {}
        with ThreadPoolExecutor(max_workers=threads) as executor:
            # Dictionary to keep track of future to key mapping
            future_to_key = {executor.submit(self.detect_with_id, text=text, scan_id=key): key for key, text in content.items()}

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
