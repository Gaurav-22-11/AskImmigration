# rag_llamaindex/reranker.py

from typing import Optional

from llama_index.core.postprocessor import SentenceTransformerRerank

# If the above import fails with ImportError, try this instead:
# from llama_index.indices.postprocessor import SentenceTransformerRerank


def get_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int = 5,
) -> SentenceTransformerRerank:
    """
    Build a SentenceTransformer cross-encoder reranker.

    - model_name: HF cross-encoder model. MiniLM is small (~400MB) and good.
    - top_n: how many nodes to keep *after* reranking.
    """
    reranker = SentenceTransformerRerank(
        model=model_name,
        top_n=top_n,
    )
    return reranker
