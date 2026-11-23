# Agentic RAG Explained

**Query rewriting, routing, and self-checking with LangGraph**

A beginner-friendly Colab walkthrough. You build one small prototype from start to finish — a naive RAG baseline, then a LangGraph agent that rewrites the question, routes to the right document store, retrieves in steps, and checks its own evidence — and compare both systems on the same data.

Companion video: *Agentic RAG Explained: Query Rewriting, Routing, and Self-Checking with LangGraph* (Agentic AI Hub).

## What you will learn

1. What agentic RAG is, with an analogy, before any retrieval code
2. A **naive RAG** baseline that cites its sources
3. **Query rewriting** — messy wording such as "pro plan" and "file cap"
4. **Routing** — pick the product, billing, or policy store
5. **Multi-step retrieval** and **self-checking** — go back if the evidence is thin
6. **Grounded answers** that cite source documents, and say "I do not know" when the evidence is thin
7. An **evaluation** against the naive baseline, using real measured numbers
8. When an agent is **overkill**

## The demo data

The notebook uses **Alderpeak**, a fictional company, and **Ticketly**, its help desk. There are 12 short documents in `data/alderpeak/`, split into three stores: product, billing, and policy. A few traps are built in on purpose, so rewriting, routing, and multi-step retrieval have real work to do.

## Results from the recorded run

These numbers were measured on the eight-question eval set in `data/eval_questions.json`. They are not estimates.

| System | Correct | Accuracy | Citations correct | Average latency |
|--------|---------|----------|-------------------|-----------------|
| Naive RAG | 6 / 8 | 75% | 7 / 8 | 1.075 s |
| Agentic RAG | 8 / 8 | 100% | 8 / 8 | 7.523 s |

The two questions naive RAG missed were a wording trap and a multi-hop question. *Why is my team hitting the file cap on the pro plan?* uses unofficial names; the official mid-tier plan is Growth, and its attachment storage pool is 50 GB. Vector search retrieved the 25 MB per-file limit instead, so the model correctly said it did not know about a plan storage pool. The agent rewrote the question in official language, routed to product and billing, and found the 50 GB pool.

*What credit can a Growth customer claim for last Monday's API outage?* needs two facts in two documents: the outage lasted 6 hours, and the policy pays 10 percent if a Priority 1 incident lasts more than 4 hours. Naive retrieval missed the incident note, so the model would not say the credit applied. The agent retrieved both documents and answered.

A data-retention question was a tie — both systems were right, and the agent was not needed. On this run the agent was also slower on average, because it takes several model calls.

## How to run it

1. Open `Agentic_RAG_Explained.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Add your Anthropic API key as a Colab Secret named `ANTHROPIC_API_KEY`, and turn on notebook access. The notebook never prints the key.
3. Run every cell.

Locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Agentic_RAG_Explained.ipynb
```

The LLM is Anthropic Claude (`claude-haiku-5-5` by default). To use a different Claude model, change the `MODEL` constant in `src/agentic_rag_demo.py`. To switch provider, change `PROVIDER` to `"openai"` and set `OPENAI_API_KEY` (and install the `openai` package). Embeddings use a local `all-MiniLM-L6-v2` model, so no separate embedding API key is required.

## Repo layout

- `Agentic_RAG_Explained.ipynb` — the walkthrough
- `src/agentic_rag_demo.py` — helpers used by the notebook
- `data/alderpeak/` — the 12 demo documents, in three stores
- `data/eval_questions.json` — the eight evaluation questions and gold checks
- `run/results/` — the measured scores from the recorded run

## Licence

Demo documents are fictional and original to this tutorial. Use the code freely for learning.
