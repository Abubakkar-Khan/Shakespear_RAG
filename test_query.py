import requests
import json

# Test the status endpoint
print("Testing /api/status...")
response = requests.get("http://localhost:5000/api/status")
status = response.json()
print(json.dumps(status, indent=2))

print("\n" + "="*60)

# Test the models endpoint
print("\nTesting /api/models...")
response = requests.get("http://localhost:5000/api/models")
models = response.json()
print(json.dumps(models, indent=2))

print("\n" + "="*60)

# Test a simple query
print("\nTesting /api/ask with 'Who is Hamlet?'...")
response = requests.post(
    "http://localhost:5000/api/ask",
    json={
        "question": "Who is Hamlet?",
        "provider": "groq",
        "model": "llama-3.1-8b-instant"
    }
)
result = response.json()
print(json.dumps(result, indent=2))
