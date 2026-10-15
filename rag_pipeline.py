"""
rag_pipeline.py

The retrieval + generation pipeline:
  1. Hybrid retrieval -- BM25 (keyword) + dense (Chroma) candidates,
     combined via reciprocal rank fusion (LangChain's EnsembleRetriever).
  2. Rerank the fused candidates with a cross-encoder (bge-reranker-base).
     This is usually the step with the biggest quality jump: bi-encoder
     retrieval is fast but imprecise near the top of the ranking, since it
     scores query and document independently; a cross-encoder scores them
     jointly and is far more accurate, just too slow to run over the whole
     corpus -- hence retrieve-wide-then-rerank-narrow.
  3. Generate a grounded answer with Claude, citing which chunk(s) each
     claim came from, refusing to answer if nothing relevant was retrieved.

Import build_pipeline() and call chain.invoke("your question"), or run
this file directly for a couple of demo queries.
"""

import argparse
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

SYSTEM_PROMPT = """You are a documentation assistant. Answer the user's question using ONLY the
provided context chunks. Each chunk is labeled with a source id like [doc3].

Rules:
- Cite the source id(s) for every claim, e.g. "Use `client.get()` [doc2]."
- If the context doesn't contain the answer, say so plainly -- do not guess
  or use outside knowledge.
- Prefer showing the actual code from the docs over paraphrasing it.
"""


def load_chunks(persist_dir: str) -> list:
    path = Path(persist_dir) / "chunks.jsonl"
    docs = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            docs.append(Document(page_content=row["page_content"], metadata=row["metadata"]))
    return docs


def format_context(docs) -> str:
    lines = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        lines.append(f"[doc{i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(lines) if lines else "(no relevant documents found)"


def build_retriever(persist_dir: str, k: int = 20, rerank_top_n: int = 5):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    dense_retriever = vectordb.as_retriever(search_kwargs={"k": k})

    corpus_docs = load_chunks(persist_dir)
    bm25_retriever = BM25Retriever.from_documents(corpus_docs)
    bm25_retriever.k = k

    fused = EnsembleRetriever(retrievers=[bm25_retriever, dense_retriever], weights=[0.4, 0.6])

    reranker_model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    reranker = CrossEncoderReranker(model=reranker_model, top_n=rerank_top_n)

    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=fused)


def build_pipeline(persist_dir: str, model_name: str = "claude-sonnet-5"):
    retriever = build_retriever(persist_dir)
    llm = ChatAnthropic(model=model_name, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    def retrieve_and_format(question: str) -> str:
        docs = retriever.invoke(question)
        return format_context(docs)

    chain = (
        {"context": retrieve_and_format, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist_dir", default="./chroma_db")
    parser.add_argument("--question", default=None)
    args = parser.parse_args()

    chain, _ = build_pipeline(args.persist_dir)

    questions = [args.question] if args.question else [
        "How do I authenticate requests to this API?",
        "What's the rate limit and how do I handle a 429 response?",
    ]
    for q in questions:
        print("=" * 80)
        print("Q:", q)
        print(chain.invoke(q))


if __name__ == "__main__":
    main()
