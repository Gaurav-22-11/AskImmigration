import json, argparse, os, re

def chunks(txt, size=800, overlap=120):
    toks=re.findall(r"\S+\s*", txt)
    i=0
    while i<len(toks):
        piece="".join(toks[i:i+size]).strip()
        if piece: yield piece
        i+=size-overlap
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out=open(args.out,"w",encoding="utf-8")
    for line in open(args.inp,encoding="utf-8"):
        d=json.loads(line); meta={k:d[k] for k in d if k!="text"}
        for i, ch in enumerate(chunks(d["text"])):
            rec={"text":ch, **meta, "chunk_id": f'{meta["id"]}-{i}'}
            out.write(json.dumps(rec,ensure_ascii=False)+"\n")
    out.close(); print("[OK] chunked ->", args.out)
