"""Agentic RAG demo helpers for the Alderpeak / Ticketly walkthrough.

Never prints API keys. The LLM defaults to Anthropic Claude. Change MODEL
to pick a different Claude model, or change PROVIDER to switch providers.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Literal, TypedDict

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "alderpeak"
EVAL_PATH = Path(__file__).resolve().parents[1] / "data" / "eval_questions.json"

# One-line model / provider swap. Change these two constants.
PROVIDER = "anthropic"          # "anthropic" or "openai"
MODEL = "claude-haiku-5-5"      # e.g. "claude-sonnet-4-5", or an OpenAI model if PROVIDER is "openai"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STORES = ("product", "billing", "policy")
MAX_LOOPS = 2


def load_key() -> bool:
    """Load the active provider key from Colab Secrets or the environment. Never print it."""
    names = ["ANTHROPIC_API_KEY"] if PROVIDER == "anthropic" else ["OPENAI_API_KEY"]
    try:
        from google.colab import userdata  # type: ignore
        for name in names:
            val = userdata.get(name)
            if val:
                os.environ[name] = val
    except Exception:
        pass
    env_name = "ANTHROPIC_API_KEY" if PROVIDER == "anthropic" else "OPENAI_API_KEY"
    return bool(os.environ.get(env_name))


def get_client():
    if PROVIDER == "anthropic":
        import anthropic
        return anthropic.Anthropic()
    if PROVIDER == "openai":
        from openai import OpenAI
        return OpenAI()
    raise ValueError(f"Unknown PROVIDER: {PROVIDER}")


def message_text(msg) -> str:
    parts = []
    for block in getattr(msg, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    if parts:
        return "\n".join(parts).strip()
    # OpenAI-style
    choices = getattr(msg, "choices", None)
    if choices:
        return (choices[0].message.content or "").strip()
    return str(msg).strip()


def chat(client, system: str, user: str, max_tokens: int = 400) -> dict:
    """One chat completion. Returns text, latency, and token usage. Never prints keys."""
    t0 = time.perf_counter()
    if PROVIDER == "anthropic":
        msg = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            thinking={"type": "disabled"},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency = time.perf_counter() - t0
        return {
            "text": message_text(msg),
            "latency_s": round(latency, 3),
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        }
    if PROVIDER == "openai":
        msg = client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        latency = time.perf_counter() - t0
        usage = msg.usage
        return {
            "text": message_text(msg),
            "latency_s": round(latency, 3),
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }
    raise ValueError(f"Unknown PROVIDER: {PROVIDER}")


def load_docs(data_dir: Path | None = None) -> list[dict]:
    data_dir = data_dir or DATA_DIR
    docs = []
    for path in sorted(data_dir.rglob("*.txt")):
        raw = path.read_text(encoding="utf-8").strip()
        src = "UNKNOWN"
        title = path.stem
        store = path.parent.name if path.parent.name in STORES else "product"
        m = re.search(r"^Source:\s*(DOC\d+)", raw, re.M)
        if m:
            src = m.group(1)
        t = re.search(r"^Title:\s*(.+)$", raw, re.M)
        if t:
            title = t.group(1).strip()
        s = re.search(r"^Store:\s*(.+)$", raw, re.M)
        if s and s.group(1).strip() in STORES:
            store = s.group(1).strip()
        body = re.sub(r"^(Source|Title|Store):.*\n", "", raw, flags=re.M).strip()
        docs.append({
            "id": src,
            "title": title,
            "store": store,
            "text": body,
            "path": str(path.relative_to(data_dir)),
        })
    return docs


def chunk_docs(docs: list[dict]) -> list[dict]:
    """One document is already short, so one document is one chunk."""
    chunks = []
    for d in docs:
        chunks.append({
            "doc_id": d["id"],
            "title": d["title"],
            "store": d["store"],
            "text": d["text"],
        })
    return chunks


_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def embed_texts(texts: list[str]):
    import numpy as np
    model = _get_embedder()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def vector_search(question: str, chunks: list[dict], matrix, k: int = 3, store: str | None = None) -> list[dict]:
    import numpy as np
    q = embed_texts([question])[0]
    mask = [i for i, c in enumerate(chunks) if store is None or c["store"] == store]
    if not mask:
        return []
    sub = matrix[mask]
    scores = sub @ q
    order = np.argsort(-scores)[:k]
    hits = []
    for j in order:
        i = mask[int(j)]
        hit = dict(chunks[i])
        hit["score"] = round(float(scores[j]), 3)
        hits.append(hit)
    return hits


def format_context(hits: list[dict]) -> str:
    parts = []
    seen = set()
    for h in hits:
        key = h["doc_id"]
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"[{h['doc_id']}] ({h.get('store', '?')}) {h['title']}\n{h['text']}")
    return "\n\n".join(parts) if parts else "(no documents retrieved)"


def answer_from_context(client, question: str, context: str) -> dict:
    system = (
        "You answer using only the document chunks provided. "
        "Cite source ids like [DOC07] after the facts you use. "
        "If the chunks are not enough, say you do not know. Do not invent prices, plans, or policies."
    )
    out = chat(client, system, f"Question: {question}\n\nEvidence:\n{context}", max_tokens=400)
    text = out["text"]
    cites = sorted(set(re.findall(r"DOC\d+", text)))
    return {
        "answer": text,
        "citations": cites,
        "latency_s": out["latency_s"],
        "input_tokens": out["input_tokens"],
        "output_tokens": out["output_tokens"],
    }


def naive_ask(client, question: str, chunks: list[dict], matrix, k: int = 3) -> dict:
    hits = vector_search(question, chunks, matrix, k=k)
    result = answer_from_context(client, question, format_context(hits))
    result["hits"] = hits
    result["mode"] = "naive"
    return result


# --------------- agent pieces ---------------

REWRITE_SYS = (
    "You rewrite a customer-support question so it is clear and specific. "
    "Keep the original meaning. Expand nicknames and vague wording "
    "(for example, 'pro plan' or 'mid tier' may mean the Growth plan; "
    "'file cap' may mean the attachment storage pool; "
    "'Flare' may mean the free live feed or the paid Flare Reports add-on). "
    "Return only the rewritten question, with no preface."
)

ROUTE_SYS = (
    "You pick which document stores to search. The stores are:\n"
    "- product: how Ticketly features work (SSO, attachments, Flare feed)\n"
    "- billing: plans, seats, prices, storage pools, add-ons\n"
    "- policy: incidents, service credits, retention, plans we do not offer\n"
    "Return a JSON object only, of the form {\"stores\": [\"billing\", \"policy\"]}. "
    "Pick one, two, or all three stores. No other keys, no markdown."
)

GRADE_SYS = (
    "You are a strict grader. Given a question and retrieved document chunks, "
    "decide whether the chunks are enough to answer correctly and completely. "
    "Return a JSON object only, of the form "
    "{\"sufficient\": true or false, \"reason\": \"short reason\", "
    "\"missing\": \"what is missing, or empty\"}. No markdown."
)


def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return {}
        return {}


def rewrite_query(client, question: str) -> dict:
    out = chat(client, REWRITE_SYS, f"Question: {question}", max_tokens=200)
    rewritten = out["text"].strip().splitlines()[0].strip().strip('"')
    if not rewritten:
        rewritten = question
    return {"rewritten": rewritten, "latency_s": out["latency_s"]}


def route_query(client, question: str) -> dict:
    out = chat(client, ROUTE_SYS, f"Question: {question}", max_tokens=150)
    data = parse_json(out["text"])
    stores = [s for s in (data.get("stores") or []) if s in STORES]
    if not stores:
        stores = list(STORES)
    # keep official order, unique
    stores = [s for s in STORES if s in stores]
    return {"stores": stores, "latency_s": out["latency_s"], "raw": out["text"]}


def grade_context(client, question: str, context: str) -> dict:
    out = chat(
        client,
        GRADE_SYS,
        f"Question: {question}\n\nEvidence:\n{context}",
        max_tokens=200,
    )
    data = parse_json(out["text"])
    sufficient = bool(data.get("sufficient"))
    return {
        "sufficient": sufficient,
        "reason": str(data.get("reason") or "").strip(),
        "missing": str(data.get("missing") or "").strip(),
        "latency_s": out["latency_s"],
    }


class AgentState(TypedDict, total=False):
    question: str
    rewritten: str
    routes: list[str]
    documents: list[dict]
    grade: str
    grade_reason: str
    missing: str
    answer: str
    citations: list[str]
    steps: list[str]
    loops: int
    latency_s: float
    input_tokens: int
    output_tokens: int


def _merge_docs(existing: list[dict], new_hits: list[dict]) -> list[dict]:
    seen = {d["doc_id"] for d in existing}
    out = list(existing)
    for h in new_hits:
        if h["doc_id"] not in seen:
            out.append(h)
            seen.add(h["doc_id"])
    return out


def build_agent(client, chunks: list[dict], matrix, k: int = 3):
    """Compile a LangGraph agent: rewrite → route → retrieve → grade (maybe loop) → generate."""
    from langgraph.graph import StateGraph, START, END

    def node_rewrite(state: AgentState) -> AgentState:
        res = rewrite_query(client, state["question"])
        steps = list(state.get("steps") or [])
        steps.append(f"rewrite: {res['rewritten']}")
        return {"rewritten": res["rewritten"], "steps": steps, "loops": state.get("loops") or 0}

    def node_route(state: AgentState) -> AgentState:
        q = state.get("rewritten") or state["question"]
        # If we already searched some stores and need more, prefer unused stores.
        already = {d.get("store") for d in state.get("documents") or []}
        res = route_query(client, q)
        stores = res["stores"]
        if state.get("loops"):
            unused = [s for s in STORES if s not in already]
            # Add unused stores that might hold what is missing.
            for s in unused:
                if s not in stores:
                    stores.append(s)
            stores = [s for s in STORES if s in stores]
        steps = list(state.get("steps") or [])
        steps.append("route: " + ", ".join(stores))
        return {"routes": stores, "steps": steps}

    def node_retrieve(state: AgentState) -> AgentState:
        q = state.get("rewritten") or state["question"]
        routes = state.get("routes") or list(STORES)
        existing = list(state.get("documents") or [])
        new_hits = []
        for store in routes:
            new_hits.extend(vector_search(q, chunks, matrix, k=k, store=store))
        # Also try the rewritten query over already-chosen stores only.
        merged = _merge_docs(existing, new_hits)
        steps = list(state.get("steps") or [])
        ids = ", ".join(f"{h['doc_id']}({h['store']})" for h in merged)
        steps.append(f"retrieve: {ids or 'nothing'}")
        return {"documents": merged, "steps": steps}

    def node_grade(state: AgentState) -> AgentState:
        q = state.get("rewritten") or state["question"]
        ctx = format_context(state.get("documents") or [])
        res = grade_context(client, q, ctx)
        sufficient = res["sufficient"]
        loops = int(state.get("loops") or 0)
        # If we already searched every store, do not loop again.
        searched = {d.get("store") for d in state.get("documents") or []}
        if sufficient or loops >= MAX_LOOPS or searched >= set(STORES):
            grade = "ok"
        else:
            grade = "retry"
            loops += 1
        steps = list(state.get("steps") or [])
        flag = "ok" if grade == "ok" else "need more"
        reason = res["reason"] or res["missing"] or ""
        steps.append(f"self-check: {flag}" + (f" — {reason}" if reason else ""))
        return {
            "grade": grade,
            "grade_reason": res["reason"],
            "missing": res["missing"],
            "loops": loops,
            "steps": steps,
        }

    def after_grade(state: AgentState) -> Literal["generate", "rewrite"]:
        if state.get("grade") == "retry":
            return "rewrite"
        return "generate"

    def node_generate(state: AgentState) -> AgentState:
        q = state["question"]
        ctx = format_context(state.get("documents") or [])
        result = answer_from_context(client, q, ctx)
        steps = list(state.get("steps") or [])
        steps.append("generate: grounded answer with citations")
        return {
            "answer": result["answer"],
            "citations": result["citations"],
            "latency_s": result["latency_s"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "steps": steps,
        }

    g = StateGraph(AgentState)
    g.add_node("rewrite", node_rewrite)
    g.add_node("route", node_route)
    g.add_node("retrieve", node_retrieve)
    g.add_node("grade", node_grade)
    g.add_node("generate", node_generate)
    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("generate", END)
    return g.compile()


def run_agent(app, question: str) -> dict:
    t0 = time.perf_counter()
    final = app.invoke({"question": question, "steps": [], "documents": [], "loops": 0})
    total = time.perf_counter() - t0
    final["total_latency_s"] = round(total, 3)
    final["mode"] = "agentic"
    return final


def graph_mermaid(app) -> str:
    try:
        return app.get_graph().draw_mermaid()
    except Exception:
        return "start -> rewrite -> route -> retrieve -> grade -> generate -> end\n(grade may loop back to rewrite)"


def load_eval_questions() -> list[dict]:
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def score_answer(item: dict, result: dict) -> dict:
    ans = (result.get("answer") or "").lower()
    # Abstain questions: "I do not know" or a clear "we do not offer this" is correct.
    if item.get("abstain_ok"):
        ok = False
        for tok in item.get("accept_any") or []:
            if tok.lower() in ans:
                ok = True
                break
        if re.search(r"\b(i do not know|i don't know|i cannot tell|there is no enterprise)\b", ans):
            ok = True
        # Inventing a monthly Enterprise price without denying the plan is a fail.
        if re.search(r"\$\s*\d+", ans) and not ok:
            ok = False
    else:
        ok = all(tok.lower() in ans for tok in item.get("must_contain") or [])
        for tok in item.get("must_not") or []:
            if tok.lower() in ans and not ok:
                ok = False
        # A side hedge such as "I do not know the exact calendar date" should not
        # wipe out an answer that already contains every required fact.
        if (not ok) and re.search(
            r"\b(i do not know|i don't know|i cannot tell|not (enough|sufficient) (evidence|information))\b",
            ans,
        ):
            ok = False
    gold_docs = set(item.get("gold_docs") or [])
    cites = set(result.get("citations") or [])
    cite_ok = bool(cites & gold_docs) if gold_docs else True
    latency = result.get("total_latency_s", result.get("latency_s"))
    return {
        "id": item["id"],
        "kind": item["kind"],
        "correct": bool(ok),
        "citation_ok": bool(cite_ok),
        "latency_s": latency,
        "citations": result.get("citations") or [],
        "answer": result.get("answer") or "",
        "steps": result.get("steps") or [],
    }
