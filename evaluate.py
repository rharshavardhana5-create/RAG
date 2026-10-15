"""
evaluate.py

Compares two retrieval configurations on your indexed corpus:
  - naive:    dense-only top-k, no reranking
  - advanced: hybrid BM25+dense retrieval, reranked with a cross-encoder

For each configuration we measure:
  1. Retrieval hit rate -- did the chunk a question was generated from
     actually show up in the retrieved set?
  2. Faithfulness -- LLM-judge score for whether every claim in the
     generated answer is actually supported by the retrieved context.
  3. Relevance -- LLM-judge score for whether the answer addresses the
     question at all.

The eval set is bootstrapped automatically: we sample chunks from your
indexed corpus and ask Claude to write one specific question each chunk
answers. This is a standard way to get a first eval set without manual
labeling -- worth naming as such in an interview, since it does carry a
bias risk (the question is often phrased close to the source text, which
can flatter retrieval quality versus a real user's phrasing).

Usage:
    ANTHROPIC_API_KEY=sk-... python evaluate.py --persist_dir ./chroma_db --n_questions 20
"""

import argparse
import json
import random
import statistics as stats

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.prompts import ChatPromptTemplate

from rag_pipeline import EMBEDDING_MODEL, RERANKER_MODEL, SYSTEM_PROMPT, format_context, load_chunks

QUESTION_GEN_PROMPT = """Here is a chunk from a piece of technical documentation:

---
{chunk}
---

Write ONE specific question that this chunk answers, that a developer
reading the docs might actually ask. The question should require this
specific chunk to answer well -- not something answerable from general
knowledge. Respond with ONLY the question, no other text.
"""

FAITHFULNESS_PROMPT = """Context given to the assistant:
{context}

Assistant's answer:
{answer}

Does the answer contain any claim that is NOT supported by the context above?
Score faithfulness from 1 (answer makes things up / contradicts context) to
5 (every claim is directly supported by the context).
Respond with ONLY a JSON object: {{"faithfulness": <int>, "rationale": "<one sentence>"}}
"""

RELEVANCE_PROMPT = """Question: {question}

Answer: {answer}

Score how well the answer actually addresses the question, from 1 (doesn't
address it) to 5 (directly and completely addresses it).
Respond with ONLY a JSON object: {{"relevance": <int>, "rationale": "<one sentence>"}}
"""


def generate_eval_set(chunks, n_questions, client, model="claude-haiku-4-5-20251001"):
    sample = random.sample(chunks, min(n_questions, len(chunks)))
    eval_set = []
    for chunk in sample:
        resp = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": QUESTION_GEN_PROMPT.format(chunk=chunk.page_content[:1500])}],
        )
        question = resp.content[0].text.strip()
        eval_set.append({
            "question": question,
            "gold_source": chunk.metadata.get("source"),
        })
    return eval_set


def judge(client, prompt, model="claude-haiku-4-5-20251001"):
    resp = client.messages.create(model=model, max_tokens=150, messages=[{"role": "user", "content": prompt}])
    try:
        return json.loads(resp.content[0].text.strip())
    except json.JSONDecodeError:
        return {}


def run_config(label, retriever, llm, eval_set, client):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    hits = 0
    faithfulness_scores = []
    relevance_scores = []

    for item in eval_set:
        docs = retriever.invoke(item["question"])
        retrieved_sources = {d.metadata.get("source") for d in docs}
        if item["gold_source"] in retrieved_sources:
            hits += 1

        context = format_context(docs)
        messages = prompt_template.format_messages(context=context, question=item["question"])
        answer = llm.invoke(messages).content

        f = judge(client, FAITHFULNESS_PROMPT.format(context=context[:3000], answer=answer))
        r = judge(client, RELEVANCE_PROMPT.format(question=item["question"], answer=answer))
        if "faithfulness" in f:
            faithfulness_scores.append(f["faithfulness"])
        if "relevance" in r:
            relevance_scores.append(r["relevance"])

    n = len(eval_set)
    print(f"\n### {label} ###")
    print(f"  Retrieval hit rate: {hits}/{n} ({hits / n:.1%})")
    if faithfulness_scores:
        print(f"  Faithfulness (1-5): {stats.mean(faithfulness_scores):.2f}")
    if relevance_scores:
        print(f"  Relevance (1-5):    {stats.mean(relevance_scores):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist_dir", default="./chroma_db")
    parser.add_argument("--n_questions", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    client = anthropic.Anthropic()
    chunks = load_chunks(args.persist_dir)
    print(f"Loaded {len(chunks)} chunks, generating {args.n_questions} eval questions...")
    eval_set = generate_eval_set(chunks, args.n_questions, client)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=args.persist_dir, embedding_function=embeddings)
    llm = ChatAnthropic(model="claude-sonnet-5", temperature=0)

    # naive: dense-only, no rerank
    naive_retriever = vectordb.as_retriever(search_kwargs={"k": args.k})
    run_config("Naive (dense-only, top-k)", naive_retriever, llm, eval_set, client)

    # advanced: hybrid + rerank
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 20
    dense_wide = vectordb.as_retriever(search_kwargs={"k": 20})
    fused = EnsembleRetriever(retrievers=[bm25, dense_wide], weights=[0.4, 0.6])
    reranker_model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    reranker = CrossEncoderReranker(model=reranker_model, top_n=args.k)
    advanced_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=fused)
    run_config("Advanced (hybrid + rerank)", advanced_retriever, llm, eval_set, client)


if __name__ == "__main__":
    main()
