"""NLP deep check - Q1 and Q2 historical price check."""
import urllib.request
import json
import re

BASE = "http://localhost:8000/api"


def post(url, data):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        r = urllib.request.urlopen(req, timeout=120)
        return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


_, sess = post(f"{BASE}/sessions", {"region": "NSW1"})
sid = sess["id"]
print("Session:", sid)

queries = [
    ("Q1", "What was the average NSW price in July 2023?", 83.76),
    ("Q2", "What was the NSW spot price in October 2024?", 77.54),
    ("Q3", "Show me the price trend from 2023 to 2024", None),
]

for label, q, expected_price in queries:
    _, r = post(f"{BASE}/sessions/{sid}/query", {"text": q})
    if not isinstance(r, dict):
        print(f"{label} ERROR: {str(r)[:150]}")
        continue

    intent = r.get("intent", "?")
    dec = r.get("decomposition") or {}
    lb = dec.get("lookback_days", "?")
    v = r.get("verdict") or {}
    secs = v.get("answer_sections") or []
    why = v.get("why_plain_english", "")

    # Search full response text (not just why) for expected price
    full_text = json.dumps(r)
    has_expected = expected_price and str(expected_price) in full_text
    sps = any("specific_period" in json.dumps(s).lower() for s in secs)

    # Extract all dollar figures from full response
    all_dlrs = re.findall(r"\$[\d.,]+/MWh", full_text)
    unique_dlrs = list(dict.fromkeys(all_dlrs))

    print(f"\n{label}: intent={intent} lb={lb}")
    print(f"  has_expected_price_{expected_price}={has_expected}")
    print(f"  has_specific_period_stats_section={sps}")
    print(f"  all_dollar_figures={unique_dlrs[:10]}")
    print(f"  section_titles={[s.get('type', s.get('title','?')) for s in secs]}")

    # Check if period_query or date range in decomposition
    for k, v2 in dec.items():
        if v2 and k in ("period_query", "date_from", "date_to", "lookback_days", "intent", "region"):
            print(f"  decomp.{k}={v2}")
