"""NLP verification test - 5 queries."""
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
    "What was the average NSW price in July 2023?",
    "What was the NSW spot price in October 2024?",
    "Show me the price trend from 2023 to 2024",
    "What is the current NSW dispatch price?",
    "What does the federal budget energy spending mean for NEM prices?",
]

for i, q in enumerate(queries, 1):
    _, r = post(f"{BASE}/sessions/{sid}/query", {"text": q})
    if isinstance(r, dict):
        intent = r.get("intent", "?")
        dec = r.get("decomposition") or {}
        lb = dec.get("lookback_days", "?")
        v = r.get("verdict") or {}
        secs = v.get("answer_sections") or []
        stypes = [s.get("type", s.get("title", "?")) for s in secs]
        why = v.get("why_plain_english", "")
        # Extract dollar amounts like $83.76/MWh
        dlrs = re.findall(r"\$[\d.,]+/MWh", why)
        sps = any("specific_period" in str(s).lower() for s in secs)
        print(
            f"Q{i}: intent={intent} lb={lb} sps_section={sps} "
            f"dollar_mentions={dlrs[:5]}"
        )
        print(f"  section_types={stypes}")
        # For Q1/Q2 also dump first 300 chars of why
        if i <= 2:
            print(f"  why_snippet={why[:300]}")
    else:
        print(f"Q{i} ERROR: {str(r)[:150]}")
    print()
