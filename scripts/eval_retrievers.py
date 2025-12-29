from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import sys

import numpy as np

# ----- Make sure project root is on sys.path -----
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Reuse your query helpers and global settings
from rag_llamaindex import query as rag_q  # noqa: E402

DEFAULT_EVAL_PATH = ROOT_DIR / "data" / "eval" / "eval.jsonl"


# ---------- Data structures ----------

@dataclass
class EvalItem:
    qid: str
    question: str
    relevant_urls: List[str]


# ---------- Loading eval set ----------

def load_eval(path: Path) -> List[EvalItem]:
    """
    Expected format per line in eval.jsonl (preferred):
      {
        "id": "q1",
        "question": "...",
        "relevant_urls": ["https://...", "https://..."]
      }

    Backwards-compatibility:
      - accepts "relevant" or "relevant_url" or "gold_urls"
      - if someone still has "relevant_ids", we will load them into relevant_urls
        ONLY IF they look like URLs. (Otherwise your eval would be meaningless.)
    """
    items: List[EvalItem] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            qid = str(rec.get("id") or rec.get("qid"))
            question = rec["question"]

            # Preferred key
            rel = (
                rec.get("relevant_urls")
                or rec.get("gold_urls")
                or rec.get("relevant_url")
                or rec.get("relevant")
                or []
            )

            # Fallback: if they used relevant_ids but accidentally stored URLs there
            if not rel and rec.get("relevant_ids"):
                maybe = rec["relevant_ids"]
                if isinstance(maybe, str):
                    maybe = [maybe]
                # Keep only URL-like entries
                rel = [x for x in maybe if isinstance(x, str) and x.startswith("http")]

            if isinstance(rel, str):
                rel = [rel]

            # Normalize: strip whitespace, drop empties
            rel_urls = [u.strip() for u in rel if isinstance(u, str) and u.strip()]
            items.append(EvalItem(qid=qid, question=question, relevant_urls=rel_urls))

    print(f"[Eval] Loaded {len(items)} eval items from {path}")
    return items


# ---------- Build retrievers (reuse query.py helpers) ----------

def build_retrievers(k_dense: int = 10):
    """
    Use the same corpus + models as query.py:
      - _load_nodes
      - _load_bm25
      - _load_dense
    """
    nodes = rag_q._load_nodes()
    bm25 = rag_q._load_bm25(nodes)
    dense = rag_q._load_dense(nodes)

    # If retriever supports changing top_k, set it
    if hasattr(dense, "similarity_top_k"):
        dense.similarity_top_k = k_dense

    return nodes, bm25, dense


# ---------- Helpers: node -> URL ----------

def _node_to_url(node) -> Optional[str]:
    """
    Pull URL from metadata. Return None if missing.
    """
    meta = getattr(node, "metadata", None) or {}
    url = meta.get("url")
    if not url:
        return None
    return str(url).strip() or None


# ---------- Retrieval wrappers (return ranked URLs) ----------

def retrieve_bm25_urls(bm25, question: str, k: int) -> List[str]:
    hits = bm25.retrieve(question)

    print("\n[DEBUG] First BM25 hit node fields:")
    if hits:
        n = hits[0].node
        print("node.id_        =", getattr(n, "id_", None))
        print("node.node_id    =", getattr(n, "node_id", None))
        print("node.ref_doc_id =", getattr(n, "ref_doc_id", None))
        print("metadata keys   =", list((getattr(n, "metadata", None) or {}).keys()))
        print("metadata        =", (getattr(n, "metadata", None) or {}))
        print("resolved url    =", _node_to_url(n))

    urls: List[str] = []
    for h in hits[:k]:
        u = _node_to_url(h.node)
        if u:
            urls.append(u)
    return urls


def retrieve_dense_urls(dense, question: str, k: int) -> List[str]:
    hits = dense.retrieve(question)
    urls: List[str] = []
    for h in hits[:k]:
        u = _node_to_url(h.node)
        if u:
            urls.append(u)
    return urls


def retrieve_hybrid_urls(bm25, dense, question: str, k: int) -> List[str]:
    """
    Simple hybrid: BM25 + dense scores summed over shared URLs.
    (Doc-level fusion; perfect match for URL-based evaluation.)
    """
    bm25_hits = bm25.retrieve(question)
    dense_hits = dense.retrieve(question)

    combined: Dict[str, float] = {}

    for h in bm25_hits:
        u = _node_to_url(h.node)
        if not u:
            continue
        combined[u] = combined.get(u, 0.0) + float(h.score)

    for h in dense_hits:
        u = _node_to_url(h.node)
        if not u:
            continue
        combined[u] = combined.get(u, 0.0) + float(h.score)

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [u for u, _ in ranked[:k]]


# ---------- Metrics ----------

def evaluate_method(
    name: str,
    eval_items: List[EvalItem],
    k: int,
    retriever_func,
) -> Dict[str, float]:
    recalls = []
    mrrs = []
    ndcgs = []

    for item in eval_items:
        gold = set(item.relevant_urls)
        if not gold:
            # If you have no gold URLs for a question, skip it (or count as 0).
            continue

        retrieved = retriever_func(item.question, k) or []

        if not retrieved:
            recalls.append(0.0)
            mrrs.append(0.0)
            ndcgs.append(0.0)
            continue

        # Binary relevance vector (doc-level)
        hits = [1 if url in gold else 0 for url in retrieved[:k]]

        # Recall@k (doc-level): did we retrieve any gold docs?
        # Common choice: (# of unique gold docs retrieved) / (# gold docs)
        retrieved_unique = set(retrieved[:k])
        recalls.append(len(retrieved_unique.intersection(gold)) / len(gold))

        # MRR@k
        mrr = 0.0
        for rank, h in enumerate(hits, start=1):
            if h:
                mrr = 1.0 / rank
                break
        mrrs.append(mrr)

        # nDCG@k (binary)
        dcg = 0.0
        for rank, h in enumerate(hits, start=1):
            if h:
                dcg += 1.0 / np.log2(rank + 1)

        # Ideal DCG: all 1s up to min(len(gold), k)
        ideal_ones = min(len(gold), k)
        idcg = 0.0
        for rank in range(1, ideal_ones + 1):
            idcg += 1.0 / np.log2(rank + 1)

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    def safe_mean(x: List[float]) -> float:
        return float(np.mean(x)) if x else 0.0

    metrics = {
        "recall": safe_mean(recalls),
        "mrr": safe_mean(mrrs),
        "ndcg": safe_mean(ndcgs),
    }

    print(
        f"[{name}] "
        f"Recall@{k}: {metrics['recall']:.3f}  "
        f"MRR@{k}: {metrics['mrr']:.3f}  "
        f"nDCG@{k}: {metrics['ndcg']:.3f}"
    )
    return metrics


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-file", type=str, default=str(DEFAULT_EVAL_PATH))
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    eval_path = Path(args.eval_file)
    eval_items = load_eval(eval_path)

    print("[Load] BM25 + dense retrievers...")
    _, bm25, dense = build_retrievers(k_dense=args.k)

    print("\n=== Retriever Evaluation (URL-level) ===")
    bm25_metrics = evaluate_method(
        "BM25",
        eval_items,
        args.k,
        lambda q, k: retrieve_bm25_urls(bm25, q, k),
    )

    dense_metrics = evaluate_method(
        "Dense",
        eval_items,
        args.k,
        lambda q, k: retrieve_dense_urls(dense, q, k),
    )

    hybrid_metrics = evaluate_method(
        "Hybrid",
        eval_items,
        args.k,
        lambda q, k: retrieve_hybrid_urls(bm25, dense, q, k),
    )

    print("\nSummary:")
    for name, m in [
        ("BM25", bm25_metrics),
        ("Dense", dense_metrics),
        ("Hybrid", hybrid_metrics),
    ]:
        print(
            f"  {name:6s}  "
            f"Recall@{args.k}: {m['recall']:.3f},  "
            f"MRR@{args.k}: {m['mrr']:.3f},  "
            f"nDCG@{args.k}: {m['ndcg']:.3f}"
        )


if __name__ == "__main__":
    main()
