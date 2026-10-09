import requests
import json
import time

print("Reloading data...")
response = requests.post("http://localhost:5000/api/reload")
result = response.json()
print(json.dumps(result, indent=2))

time.sleep(2)

print("\nChecking status...")
response = requests.get("http://localhost:5000/api/status")
status = response.json()
print(json.dumps(status, indent=2))
