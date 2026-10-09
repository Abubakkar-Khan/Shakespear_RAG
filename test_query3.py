import requests
import json

# Test a query with the configured Llama 8B model from .env
print("Testing /api/ask with 'Who is Caius Marcius?' (using Groq Llama 8B)...")
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
