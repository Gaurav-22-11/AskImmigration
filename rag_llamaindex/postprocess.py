from typing import List
from sentence_transformers import CrossEncoder
import numpy as np
from rag_llamaindex.settings import CROSS_ENC, NLI_MODEL, NLI_THRESHOLD

class CrossEncoderReranker:
    def __init__(self, model_name=CROSS_ENC, top_k=10):
        self.m = CrossEncoder(model_name); self.top_k=top_k
    def __call__(self, q:str, texts:List[str]):
        pairs=[(q,t) for t in texts]; s=self.m.predict(pairs, convert_to_numpy=True)
        order=np.argsort(-s)[:self.top_k]
        return [texts[i] for i in order], s[order], order

class NliVerifier:
    def __init__(self, model_name=NLI_MODEL, thr=NLI_THRESHOLD):
        self.m=CrossEncoder(model_name); self.thr=thr
    def verify(self, context:str, answer:str):
        logits=self.m.predict([(context,answer)], convert_to_numpy=True)[0]
        p=np.exp(logits-logits.max()); p/=p.sum()
        entail=float(p[-1]); return entail>=self.thr, entail
