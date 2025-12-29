import json
from pathlib import Path

path = Path("data/processed/corpus.jsonl")
ids = set()
with path.open(encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line)
        ids.add(rec['id'])
print("Total IDs:", len(ids))
print("Sample:", list(ids)[:10])