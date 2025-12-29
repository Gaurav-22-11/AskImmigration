# scripts/ask.py
import argparse
import sys
from pathlib import Path

# --- Ensure project root is on sys.path ---
ROOT = Path(__file__).resolve().parents[1]  # .../Immi
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_llamaindex.query import query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question",
        type=str,
        help="User question about visas/immigration",
    )
    args = parser.parse_args()

    ans, score, legend = query(args.question)

    print(f"\nQ: {args.question}\n")
    print("Answer:\n", ans)
    print("\n[Verification score]:", round(score, 3))
    print("\nSources:")
    for i, u in legend:
        print(f"[{i}] {u}")


if __name__ == "__main__":
    main()
