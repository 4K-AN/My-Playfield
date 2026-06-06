import requests

data = {
    "YearsExperience": 5,
    "Education": "Bachelor",
    "Role": "Software Engineer",
    "Location": "New York"
}

try:
    response = requests.post("http://localhost:8000/predict", json=data)
    print(response.json())
except Exception as e:
    print(e)
