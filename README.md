# DocsRAG — Hybrid Retrieval RAG over Technical Documentation

A retrieval-augmented question-answering system over API/technical docs,
built to demonstrate production RAG techniques rather than a naive
"embed everything, take top-k" demo.

## Why this project

Most RAG tutorials stop at dense-only top-k retrieval and call it done.
This one measures where that actually falls short and fixes it: hybrid
retrieval (catches exact API names and error codes that embeddings alone
miss), cross-encoder reranking, code-aware chunking, and a real
naive-vs-advanced evaluation instead of a single demo query.

## Project structure

| File | Purpose |
|---|---|
| `ingest.py` | Load docs (local files or URLs), code-aware chunking, embed, persist to Chroma |
| `rag_pipeline.py` | Hybrid retrieval (BM25 + dense) → cross-encoder rerank → grounded generation |
| `evaluate.py` | Naive vs. advanced retrieval comparison: hit rate, faithfulness, relevance |
| `cli.py` | Interactive terminal Q&A |
| `requirements.txt` | Dependencies |

## How it works

1. **Ingest** (`ingest.py`) — reads your docs, splits them into ~800-char
   chunks *without* cutting fenced code blocks in half, embeds each chunk
   with `BAAI/bge-small-en-v1.5` (small, fast, runs on CPU — no API key or
   GPU needed for this step), and persists everything to a local Chroma
   database.
2. **Retrieve** (`rag_pipeline.py`) — runs BM25 keyword search and dense
   vector search in parallel, fuses them with reciprocal rank fusion
   (`EnsembleRetriever`), then reranks the fused candidates with a
   cross-encoder (`BAAI/bge-reranker-base`) before keeping only the top 5.
3. **Generate** — the top chunks go to Claude with a system prompt that
   requires citing a source id per claim and refusing to answer from
   outside knowledge if nothing relevant was retrieved.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Try it on a real API docs set

```bash
git clone --depth 1 https://github.com/tiangolo/fastapi
python ingest.py --source_dir fastapi/docs/en/docs --persist_dir ./chroma_db
python cli.py --persist_dir ./chroma_db
```

Or point `--urls` at any public docs pages instead of a local clone.

## Evaluate

```bash
python evaluate.py --persist_dir ./chroma_db --n_questions 20
```

This bootstraps 20 questions from your own indexed chunks (asking Claude
to write a question each chunk answers), then runs both retrieval
configurations against every question.

## Results

*(Fill in after your run.)*

| Metric | Naive (dense-only) | Advanced (hybrid + rerank) |
|---|---|---|
| Retrieval hit rate | | |
| Faithfulness (1–5) | | |
| Relevance (1–5) | | |

## Talking points for interviews

- **Why hybrid retrieval, not just dense** — dense embeddings capture
  semantic similarity but often miss exact tokens: a function name like
  `client.get_async()` or an error code like `429` can get diluted in an
  embedding, while BM25 finds it directly. Fusing both catches more.
- **Why rerank instead of just increasing k** — a cross-encoder scores
  query and document *jointly*, which is far more accurate than a
  bi-encoder's independent scoring, but too slow to run over an entire
  corpus. Retrieve wide (20) with cheap bi-encoders, then rerank narrow
  (5) with the expensive-but-accurate cross-encoder — the standard
  two-stage retrieval pattern.
- **Why code-aware chunking** — a naive character-count splitter will
  happily cut a code example in half, which is actively worse than
  omitting it for an API-docs assistant. Protecting fenced code blocks
  before splitting is a small detail that measurably changes answer
  quality on code-heavy docs.
- **Why citations are enforced in the prompt** — an ungrounded RAG answer
  is a support ticket waiting to happen. Forcing per-claim citations to a
  labeled chunk both reduces hallucination and gives the user (or an
  eval harness) something to check the answer against.
- **What you'd do next** — query decomposition for multi-hop questions
  ("how do I do X and then Y"), a feedback loop that flags low-confidence
  answers for human review, or extending retrieval to include the actual
  API's OpenAPI spec alongside prose docs.

## Known limitations

- Bootstrapped eval questions are generated from the same chunks they're
  scored against, which can flatter retrieval — a held-out, human-written
  question set would be more rigorous.
- No cross-document reasoning: each answer is grounded in independently
  retrieved chunks, not a synthesized understanding of the whole corpus.
- The embedding and reranker models are English-optimized; multilingual
  docs would need different model choices.
