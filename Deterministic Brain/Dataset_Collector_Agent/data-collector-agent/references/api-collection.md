# API Collection

Structured, legal, and reliable — prefer this over scraping whenever an API exists.

## API types

| Type | Description |
|---|---|
| REST | Most common, uses HTTP methods (GET, POST) |
| GraphQL | Client specifies exactly which fields to return |
| WebSocket | Real-time bidirectional stream |
| SOAP | Older, XML-based, common in enterprise/banking |

## Basic REST call

```python
import requests
import pandas as pd

url = "https://restcountries.com/v3.1/all"
response = requests.get(url)
print(response.status_code)  # 200 = success

data = response.json()
records = [{
    "name": c["name"]["common"],
    "capital": c.get("capital", ["N/A"])[0],
    "population": c.get("population", 0),
    "region": c.get("region", "N/A"),
} for c in data]

df = pd.DataFrame(records)
df.to_csv("countries.csv", index=False)
```

## With API key authentication

```python
import requests, os

API_KEY = os.environ["WEATHER_API_KEY"]  # never hardcode keys
url = "https://api.openweathermap.org/data/2.5/weather"
params = {"q": "Pune", "appid": API_KEY, "units": "metric"}

response = requests.get(url, params=params, timeout=10)
data = response.json()
```

## Pagination

```python
import requests, pandas as pd

all_data = []
page = 1
while True:
    resp = requests.get(
        "https://api.example.com/data",
        params={"page": page, "per_page": 100},
        timeout=10,
    )
    data = resp.json()
    if not data["results"]:
        break
    all_data.extend(data["results"])
    page += 1

df = pd.DataFrame(all_data)
```

## HTTP status codes you must handle

| Code | Meaning | Action |
|---|---|---|
| 200 | Success | Process data |
| 400 | Bad request | Fix parameters |
| 401 | Unauthorized | Check API key |
| 403 | Forbidden | No access |
| 404 | Not found | Wrong URL |
| 429 | Too many requests | Back off, reduce rate |
| 500 | Server error | Retry later |

Wrap every call with the retry/backoff pattern in `code-generation.md` — 429 and 500 specifically should trigger exponential backoff, not an immediate hard failure.
