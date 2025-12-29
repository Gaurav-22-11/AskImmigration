from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ENCODER_NAME = "BAAI/bge-small-en-v1.5"
GEN_NAME     = "google/flan-t5-base"
CROSS_ENC    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL    = "cross-encoder/nli-deberta-v3-small"
NLI_THRESHOLD = 0.6

EMBEDDING = HuggingFaceEmbedding(model_name=ENCODER_NAME)

def get_hf_llm():
    tok = AutoTokenizer.from_pretrained(GEN_NAME)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(GEN_NAME)
    return tok, mdl

Settings.embed_model = EMBEDDING
Settings.chunk_size = 800
Settings.chunk_overlap = 120
