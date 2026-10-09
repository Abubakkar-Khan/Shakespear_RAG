import requests
import json

# Test a query that should match the content
print("Testing /api/ask with 'Who is Caius Marcius?'...")
response = requests.post(
    "http://localhost:5000/api/ask",
    json={
        "question": "Who is Caius Marcius?",
        "provider": "groq",
        "model": "llama-3.1-8b-instant"
    }
)
result = response.json()
print(json.dumps(result, indent=2))

print("\n" + "="*60)

# Test another query
print("\nTesting /api/ask with 'What do the citizens want?'...")
response = requests.post(
    "http://localhost:5000/api/ask",
    json={
        "question": "What do the citizens want?",
        "provider": "groq",
        "model": "llama-3.1-8b-instant"
    }
)
result = response.json()
print(json.dumps(result, indent=2))
