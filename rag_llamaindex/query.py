from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import os
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import TextNode

# Modern llama-index import
try:
    from llama_index.core import VectorStoreIndex, Settings
except ImportError:
    from llama_index import VectorStoreIndex, Settings

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank

from sentence_transformers import CrossEncoder
import google.generativeai as genai


# ---------- Paths & model config ----------
ROOT_DIR = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT_DIR / "data" / "processed" / "corpus.jsonl"

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

NLI_THRESHOLD = 0.8

# Gemini
GEMINI_MODEL_NAME = "gemini-2.5-flash"  # or "gemini-1.5-flash" if you prefer


# ---------- Global models ----------
# Embeddings for dense retrieval
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

# NLI verifier
_nli = CrossEncoder(NLI_MODEL_NAME)

# Reranker (cross-encoder)
_reranker = SentenceTransformerRerank(
    model=RERANK_MODEL,
    top_n=5,  # keep 5 best chunks
)

# Gemini setup
_gemini_api_key = os.getenv("GEMINI_API_KEY")
if not _gemini_api_key:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable not set. "
        "Create a Gemini key in Google AI Studio and export GEMINI_API_KEY."
    )

genai.configure(api_key=_gemini_api_key)
_gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)


# ---------- Helpers ----------
def _load_nodes() -> List[TextNode]:
    """Load TextNodes from corpus.jsonl with stable IDs."""
    nodes: List[TextNode] = []
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec.get("text", "").strip()
            if not text:
                continue

            chunk_id = str(rec.get("id") or rec.get("chunk_id") or "")
            if not chunk_id:
                raise ValueError("Each corpus record must have an 'id' (stable chunk id).")

            meta = {k: rec[k] for k in ("url", "agency", "title") if k in rec}
            meta["chunk_id"] = chunk_id  # keep it in metadata too

            # IMPORTANT: give TextNode a stable id
            nodes.append(TextNode(id_=chunk_id, text=text, metadata=meta))

    print(f"[Corpus] Loaded {len(nodes)} nodes from {CORPUS_PATH}")
    return nodes


def _load_bm25(nodes: List[TextNode]):
    """BM25 retriever over the nodes."""
    return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)


def _load_dense(nodes: List[TextNode]):
    """Dense retriever using the configured embed model."""
    index = VectorStoreIndex(nodes)
    return index.as_retriever(similarity_top_k=10)


def _nli_verify(answer: str, context: str) -> float:
    """
    Use NLI CrossEncoder to compute entailment probability
    that context -> answer.
    """
    logits = _nli.predict([(context, answer)], convert_to_numpy=True)
    logits = np.asarray(logits, dtype="float32").reshape(1, -1)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    # index 2 corresponds to 'entailment' in most NLI heads
    return float(probs[0, 2])


def _generate_answer_with_gemini(question: str, context: str) -> str:
    """
    Use Gemini to generate an answer strictly based on the provided context.
    """
    prompt = (
        "You are an immigration assistant. Answer the user's question ONLY using "
        "the provided context from official U.S. government sources. "
        "If the answer is not clearly supported by the context, say that you "
        "cannot answer with certainty.\n\n"
        "Context:\n"
        f"{context}\n\n"
        "User question:\n"
        f"{question}\n\n"
        "Answer in clear, concise English. Start with a short direct answer, "
        "then provide a brief explanation."
    )

    resp = _gemini_model.generate_content(prompt)
    # Simple usage: resp.text
    return (resp.text or "").strip()


# ---------- Main Query Function ----------
def query(question: str) -> Tuple[str, float, List[Tuple[int, str]]]:
    """
    Main RAG pipeline:
      1. Load corpus
      2. Hybrid retrieval (BM25 + dense)
      3. Cross-encoder reranking
      4. Answer generation with Gemini using top chunks
      5. NLI verification score
      6. Legend of source URLs
    """

    # LOAD CORPUS ---------------------------------------------------
    nodes = _load_nodes()

    # RETRIEVAL -----------------------------------------------------
    bm25 = _load_bm25(nodes)
    dense = _load_dense(nodes)

    bm25_hits = bm25.retrieve(question)   # list[NodeWithScore]
    dense_hits = dense.retrieve(question) # list[NodeWithScore]

    # Hybrid score fusion: sum normalized scores by node_id
    combined: dict[str, list] = {}
    for hit in bm25_hits:
        combined[hit.node.node_id] = [hit, hit.score]

    for hit in dense_hits:
        if hit.node.node_id in combined:
            combined[hit.node.node_id][1] += hit.score
        else:
            combined[hit.node.node_id] = [hit, hit.score]

    # Sort by fused score (descending)
    hybrid_ranked = sorted(combined.values(), key=lambda x: x[1], reverse=True)
    hybrid_hits = [h[0] for h in hybrid_ranked]  # NodeWithScore objects

    # RERANKING -----------------------------------------------------
    # Take top 20 from hybrid and rerank with cross-encoder
    initial_candidates = hybrid_hits[:20]

    reranked_hits = _reranker.postprocess_nodes(
        initial_candidates,
        query_str=question,
    )

    # Keep top 5 final nodes (TextNode objects)
    top_nodes = [hit.node for hit in reranked_hits[:5]]

    # ANSWER GENERATION ---------------------------------------------
    context = "\n\n".join(n.text for n in top_nodes)

    generated_answer = _generate_answer_with_gemini(question, context)

    # Optionally append raw excerpts below the generated answer
    answer = (
        generated_answer
        + "\n\nHere are the most relevant excerpts from official sources:\n\n"
        + context
    )

    # VERIFICATION --------------------------------------------------
    verification = _nli_verify(generated_answer, context)

    # LEGEND --------------------------------------------------------
    legend: List[Tuple[int, str]] = []
    seen: set[str] = set()
    i = 1
    for n in top_nodes:
        url = n.metadata.get("url") if n.metadata else None
        if url and url not in seen:
            legend.append((i, url))
            seen.add(url)
            i += 1

    return answer, verification, legend
