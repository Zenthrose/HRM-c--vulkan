import requests
import json
import os

API_KEY = os.getenv("XAI_API_KEY", "YOUR_API_KEY_HERE")  # Set environment variable
BASE_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-4-latest"

def grok_query(prompt, system="You are a Vulkan AI engineer. Fix code errors precisely."):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "model": MODEL,
        "stream": False,
        "temperature": 0.1
    }
    response = requests.post(BASE_URL, headers=headers, json=payload)
    if response.status_code == 429:
        print("Rate limit hit—wait 60s and retry.")
        return None
    return response.json()["choices"][0]["message"]["content"]

# Test it
if __name__ == "__main__":
    test = grok_query("Say hello from Grok 4.1—no errors.")
    print(test)