# app/main.py

from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# --- Ensure project root is on sys.path (same trick as scripts/ask.py) ---
ROOT = Path(__file__).resolve().parents[1]  # .../Immi
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_llamaindex.query import query  # your existing RAG+Gemini pipeline


app = FastAPI(title="AskImmigration RAG Demo")


class Question(BaseModel):
    question: str


# ---------- Simple HTML UI at "/" ----------
@app.get("/", response_class=HTMLResponse)
async def index():
    # Very small single-page UI
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>AskImmigration – Demo</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f172a;
      color: #e5e7eb;
      display: flex;
      justify-content: center;
      padding: 2rem 1rem;
    }
    .container {
      width: 100%;
      max-width: 900px;
      background: #020617;
      border-radius: 1rem;
      padding: 1.5rem;
      box-shadow: 0 20px 40px rgba(0,0,0,0.5);
      border: 1px solid #1f2937;
    }
    h1 {
      font-size: 1.6rem;
      margin-bottom: 0.25rem;
    }
    .subtitle {
      font-size: 0.9rem;
      color: #9ca3af;
      margin-bottom: 1.5rem;
    }
    label {
      font-size: 0.85rem;
      color: #9ca3af;
      display: block;
      margin-bottom: 0.25rem;
    }
    textarea {
      width: 100%;
      min-height: 80px;
      max-height: 200px;
      padding: 0.75rem;
      border-radius: 0.75rem;
      border: 1px solid #374151;
      background: #020617;
      color: #e5e7eb;
      resize: vertical;
      font-family: inherit;
      font-size: 0.95rem;
    }
    textarea:focus {
      outline: none;
      border-color: #38bdf8;
      box-shadow: 0 0 0 1px #38bdf8;
    }
    button {
      margin-top: 0.75rem;
      padding: 0.6rem 1.2rem;
      border-radius: 999px;
      border: none;
      cursor: pointer;
      font-size: 0.95rem;
      font-weight: 500;
      background: linear-gradient(135deg, #0ea5e9, #6366f1);
      color: white;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
    }
    button:disabled {
      opacity: 0.6;
      cursor: default;
    }
    .spinner {
      width: 16px;
      height: 16px;
      border-radius: 999px;
      border: 2px solid rgba(255,255,255,0.3);
      border-top-color: white;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    .answer-card {
      margin-top: 1.5rem;
      padding: 1rem;
      border-radius: 0.75rem;
      background: #020617;
      border: 1px solid #1f2937;
      max-height: 400px;
      overflow-y: auto;
      white-space: pre-wrap;
      font-size: 0.9rem;
      line-height: 1.4;
    }
    .meta {
      margin-top: 0.75rem;
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      font-size: 0.8rem;
      color: #9ca3af;
    }
    .badge {
      padding: 0.25rem 0.6rem;
      border-radius: 999px;
      border: 1px solid #374151;
    }
    .sources {
      margin-top: 0.5rem;
      font-size: 0.8rem;
    }
    .sources a {
      color: #38bdf8;
      text-decoration: none;
    }
    .sources a:hover {
      text-decoration: underline;
    }
    .error {
      margin-top: 0.75rem;
      color: #f97373;
      font-size: 0.85rem;
    }
    .footer {
      margin-top: 1.5rem;
      font-size: 0.75rem;
      color: #6b7280;
    }
    code {
      background: #111827;
      padding: 0.1rem 0.3rem;
      border-radius: 0.25rem;
      font-size: 0.8rem;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>AskImmigration (RAG Demo)</h1>
    <div class="subtitle">
      Ask U.S. immigration / visa questions. Answers come from a curated DHS/USCIS corpus + a Gemini model.
    </div>

    <label for="question">Your question</label>
    <textarea id="question" placeholder="e.g. How long can I stay in the US after my F-1 program ends?"></textarea>
    <button id="askBtn">
      <span id="btnText">Ask</span>
      <span id="spinner" class="spinner" style="display:none;"></span>
    </button>

    <div id="error" class="error" style="display:none;"></div>

    <div class="answer-card" id="answerCard" style="display:none;">
      <div id="answerText"></div>
      <div class="meta">
        <div class="badge" id="scoreBadge"></div>
      </div>
      <div class="sources" id="sources"></div>
    </div>

    <div class="footer">
      Local demo – FastAPI + llama-index + Gemini 2.5 Flash.  
      Run with <code>uvicorn app.main:app --reload</code> and open <code>http://127.0.0.1:8000</code>.
    </div>
  </div>

  <script>
    const askBtn = document.getElementById("askBtn");
    const questionEl = document.getElementById("question");
    const answerCard = document.getElementById("answerCard");
    const answerText = document.getElementById("answerText");
    const scoreBadge = document.getElementById("scoreBadge");
    const sourcesDiv = document.getElementById("sources");
    const errorDiv = document.getElementById("error");
    const spinner = document.getElementById("spinner");
    const btnText = document.getElementById("btnText");

    async function askQuestion() {
      const q = questionEl.value.trim();
      if (!q) return;

      errorDiv.style.display = "none";
      answerCard.style.display = "none";

      askBtn.disabled = true;
      spinner.style.display = "inline-block";
      btnText.textContent = "Thinking...";

      try {
        const resp = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q })
        });

        if (!resp.ok) {
          throw new Error("Server error: " + resp.status);
        }

        const data = await resp.json();

        answerText.textContent = data.answer || "(No answer returned)";
        const score = data.verification_score !== null ? data.verification_score.toFixed(3) : "N/A";
        scoreBadge.textContent = "NLI verification score: " + score;

        if (Array.isArray(data.sources) && data.sources.length > 0) {
          const links = data.sources.map(s => {
            return `[${s.id}] <a href="${s.url}" target="_blank" rel="noopener noreferrer">${s.url}</a>`;
          });
          sourcesDiv.innerHTML = "Sources:<br>" + links.join("<br>");
        } else {
          sourcesDiv.textContent = "Sources: (none)";
        }

        answerCard.style.display = "block";
      } catch (err) {
        console.error(err);
        errorDiv.textContent = "Error: " + err.message;
        errorDiv.style.display = "block";
      } finally {
        askBtn.disabled = false;
        spinner.style.display = "none";
        btnText.textContent = "Ask";
      }
    }

    askQuestionOnEnter = (e) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        askQuestion();
      }
    };

    askBtn.addEventListener("click", askQuestion);
    questionEl.addEventListener("keydown", askQuestionOnEnter);
  </script>
</body>
</html>
    """


# ---------- JSON API at /ask ----------
@app.post("/ask")
async def ask(payload: Question):
    try:
        answer, score, legend = query(payload.question)
        sources = [
            {"id": int(idx), "url": url}
            for (idx, url) in legend
        ]
        return {
            "question": payload.question,
            "answer": answer,
            "verification_score": score,
            "sources": sources,
        }
    except Exception as e:
        # You can log this properly; for now, return a 500
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
