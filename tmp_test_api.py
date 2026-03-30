import requests
response = requests.post("http://localhost:8000/api/ask", json={
    "question": "What is the minimum attendance required?",
    "top_k": 3
})
print("RAW RESPONSE:", response.text)
try:
    print("ANSWER:", response.json().get('answer'))
except Exception as e:
    print(e)
