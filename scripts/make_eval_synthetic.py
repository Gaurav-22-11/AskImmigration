# scripts/make_eval_synthetic.py

from pathlib import Path
import json
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from rag_llamaindex import query as rag_q  # uses your existing query.py


OUT_PATH = Path("data/eval/eval.jsonl")


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ❶ Define a small set of realistic questions
    questions = [
        "How long can I stay in the US after my F-1 program ends?",
        "What is OPT and how is it different from CPT for F-1 students?",
        "When can a new F-1 student enter the US before classes start?",
        "What is SEVIS and what fee do I need to pay?"
    ]

    # ❷ Load nodes and BM25 retriever from your query module
    nodes = rag_q._load_nodes()
    bm25 = rag_q._load_bm25(nodes)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for i, q in enumerate(questions, start=1):
            hits = bm25.retrieve(q)

            # Take top 3 as "relevant" for this synthetic eval
            relevant_ids = [h.node.node_id for h in hits[:3]]

            rec = {
                "id": f"q{i}",
                "question": q,
                "relevant_ids": relevant_ids,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote synthetic eval set with {len(questions)} items to {OUT_PATH}")


if __name__ == "__main__":
    main()
