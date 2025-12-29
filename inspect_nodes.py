# scripts/inspect_nodes.py
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="data/processed/chunks.jsonl",
        help="Path to corpus or chunk file",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--contains", type=str, default=None,
                        help="Filter by substring in text (case-insensitive)")
    args = parser.parse_args()

    path = Path(args.path)
    with path.open("r", encoding="utf-8") as f:
        count = 0
        for line in f:
            d = json.loads(line)
            text = d.get("text", "")
            if args.contains:
                if args.contains.lower() not in text.lower():
                    continue
            node_id = d.get("id") or d.get("doc_id") or d.get("node_id")
            meta = d.get("metadata", {})
            url = meta.get("url") or meta.get("source") or ""
            snippet = text.replace("\n", " ")[:220]
            print("=" * 80)
            print(f"ID:   {node_id}")
            if url:
                print(f"URL:  {url}")
            print(f"TEXT: {snippet}...")
            count += 1
            if count >= args.limit:
                break

if __name__ == "__main__":
    main()
