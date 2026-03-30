import requests
import json

url = "http://localhost:8000/api/ask"
payload = {
    "question": "What is the minimum attendance required?",
    "top_k": 3,
    "max_tokens": 150,
    "temperature": 0.1
}

try:
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        with open('api_ask_result.txt', 'w', encoding='utf-8') as f:
            f.write(data["answer"])
        print("Success! Saved to api_ask_result.txt")
    else:
        print("Error:", response.status_code, response.text)
except Exception as e:
    print("Request failed:", e)
