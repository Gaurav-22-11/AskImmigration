import argparse, json, os, hashlib, re, datetime as dt
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

def sid(s): return hashlib.md5((s or "").encode()).hexdigest()[:16]
def today(): return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
def agency(url):
    h=urlparse(url).netloc
    return "USCIS" if "uscis" in h else "STATE" if "state.gov" in h else h.upper()

JS_GET = """
() => {
  function norm(s){return (s||"").replace(/\\s+/g,' ').trim();}
  const roots=[document.querySelector('main'),document.querySelector('article'),
    document.querySelector('#block-uscis-content'),document.body].filter(Boolean);
  let chosen=null; for(const r of roots){ if(r.innerText && r.innerText.trim().length>200){chosen=r;break;}}
  if(!chosen) chosen=document.body;
  const c=chosen.cloneNode(true);
  c.querySelectorAll('script,style,noscript,header,footer,nav,aside').forEach(e=>e.remove());
  c.querySelectorAll('[role=banner],[role=navigation],[role=contentinfo],.usa-banner,.cookie,.consent').forEach(e=>e.remove());
  const w=document.createTreeWalker(c, NodeFilter.SHOW_TEXT, null);
  const lines=[]; while(w.nextNode()){const t=norm(w.currentNode.nodeValue); if(t) lines.push(t);}
  const kept=[]; for(let i=0;i<lines.length;i++){ const t=lines[i]; if(i>0 && t===lines[i-1]) continue; if(t.length<3) continue; kept.push(t); }
  return kept.join(' ');
}
"""

JUNK = [r"^An official website", r"^Official websites use", r"^Secure .* HTTPS", r"^Sign In", r"\bcookie\b", r"\bprivacy\b",
        r"\bFeedback\b", r"\bMenu\b", r"\bUSA\.gov\b", r"\bNewsroom\b"]

def drop_boiler(t):
    t=re.sub(r"\s+"," ",t or "").strip()
    parts=[p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", t) if p.strip()]
    keep=[]
    for s in parts:
        if any(re.search(p,s,re.I) for p in JUNK): continue
        keep.append(s)
    out=" ".join(keep)
    return re.sub(r"\s+"," ",out).strip()

def fetch(url, wait_ms=2500):
    with sync_playwright() as p:
        b=p.chromium.launch(headless=True, args=["--no-sandbox","--disable-dev-shm-usage"])
        ctx=b.new_context(viewport={"width":1280,"height":2000},
                          user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                     "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                          extra_http_headers={"Referer":"https://www.google.com/"})
        pg=ctx.new_page(); pg.goto(url, wait_until="load", timeout=60000); pg.wait_for_timeout(wait_ms)
        for y in (1200, 3000, 6000):
            try: pg.mouse.wheel(0,y); pg.wait_for_timeout(200)
            except: pass
        title=pg.title(); text=pg.evaluate(JS_GET); html=pg.content()
        ctx.close(); b.close()
        return title,text,html

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--url-file", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs("data/raw/html_browser", exist_ok=True)

    with open(args.out,"w",encoding="utf-8") as w, open(args.url_file) as f:
        for url in [u.strip() for u in f if u.strip() and not u.startswith("#")]:
            try:
                title,dom,html=fetch(url)
                open(f"data/raw/html_browser/{sid(url)}.html","w",encoding="utf-8").write(html)
                text=drop_boiler(dom); alpha=sum(c.isalpha() for c in text)
                if alpha<200:
                    print(f"[SKIP] low-signal alpha={alpha} {url}"); continue
                rec={"id":sid(url),"url":url,"title":title or "","text":text,
                     "agency":agency(url),"last_seen":today(),"source_type":"html_browser_clean"}
                w.write(json.dumps(rec,ensure_ascii=False)+"\n")
                print("[OK]",url,f"(alpha={alpha})")
            except Exception as e:
                print("[WARN]",url,e)
