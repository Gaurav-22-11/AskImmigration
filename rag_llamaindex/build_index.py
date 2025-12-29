import json, os, faiss
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from rag_llamaindex.settings import EMBEDDING

IN_CHUNKS = "data/processed/chunks.jsonl"
FAISS_IDX = "artifacts/faiss_llamaindex.index"
BM25_STORE = "artifacts/bm25_nodes.jsonl"

def load_docs():
    docs=[]
    with open(IN_CHUNKS,"r",encoding="utf-8") as f:
        for ln in f:
            j=json.loads(ln); txt=j["text"]; meta={k:j[k] for k in j if k!="text"}
            docs.append(Document(text=txt, metadata=meta))
    return docs

if __name__=="__main__":
    os.makedirs("artifacts", exist_ok=True)
    docs=load_docs(); print(f"[INFO] docs={len(docs)}")
    # BM25 materialization
    with open(BM25_STORE,"w",encoding="utf-8") as w:
        for d in docs:
            w.write(json.dumps({"text":d.text,"metadata":d.metadata},ensure_ascii=False)+"\n")
    # FAISS
    dim = EMBEDDING._model.get_sentence_embedding_dimension()
    idx = faiss.IndexFlatIP(dim)
    vs = FaissVectorStore(faiss_index=idx)
    sc = StorageContext.from_defaults(vector_store=vs)
    VectorStoreIndex.from_documents(docs, storage_context=sc, embed_model=EMBEDDING, show_progress=True)
    def _get_faiss_index(store):
        """
        Handle different llama-index FaissVectorStore versions.

        Some expose `faiss_index`, others keep `_faiss_index` or `index`.
            This helper tries them in a safe order.
    """
    # Newer versions may store the raw index on `.index`
        if hasattr(store, "faiss_index"):
            return store.faiss_index
        if hasattr(store, "_faiss_index"):
            return store._faiss_index
        if hasattr(store, "index"):
            return store.index
        raise AttributeError(
            f"Could not find underlying FAISS index on {type(store).__name__}; "
        "check llama-index-vector-stores-faiss version or update this helper."
        )
    faiss_index = _get_faiss_index(vs)
    faiss.write_index(faiss_index, FAISS_IDX)
    print("[OK] BM25+FAISS ready")
