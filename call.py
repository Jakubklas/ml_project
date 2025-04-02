import requests
import json

url = "http://localhost:8000/classes"
payload = {
    "features": [
        [5.1, 11.4, 66.1]
    ]
}

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}

response = requests.post(url, data=json.dumps(payload), headers=headers)
print(response.json().get("predictions"))