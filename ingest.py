"""
ingest.py

Builds a local, persisted vector index over a set of technical/API
documentation pages, ready for hybrid (BM25 + dense) retrieval.

Two input modes (use either or both):
  1. Local files:  --source_dir path/to/docs   (.md, .txt, .rst, .html)
  2. URLs:         --urls url1 url2 ...         (fetched and HTML-stripped)

Chunking respects fenced code blocks (``` ... ```) -- they're protected
from being split mid-block before the text splitter runs, since a
truncated code example is worse than no example at all in an API-docs
RAG system.

Example -- point it at a real API docs set to try this out:
    git clone --depth 1 https://github.com/tiangolo/fastapi
    python ingest.py --source_dir fastapi/docs/en/docs --persist_dir ./chroma_db

Output:
    <persist_dir>/            -- the Chroma vector store
    <persist_dir>/chunks.jsonl -- the same chunks in plain text, used by
                                   rag_pipeline.py to build the BM25 index
                                   without depending on Chroma internals
"""

import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".html", ".htm"}


def load_local_files(source_dir: str) -> list:
    docs = []
    for path in Path(source_dir).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        text = path.read_text(errors="ignore")
        if path.suffix.lower() in {".html", ".htm"}:
            text = BeautifulSoup(text, "html.parser").get_text("\n")
        docs.append(Document(page_content=text, metadata={"source": str(path)}))
    return docs


def load_urls(urls: list) -> list:
    import requests

    docs = []
    for url in urls:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "rag-docs-bot/1.0"})
        resp.raise_for_status()
        text = BeautifulSoup(resp.text, "html.parser").get_text("\n")
        docs.append(Document(page_content=text, metadata={"source": url}))
    return docs


def code_aware_split(docs: list, chunk_size: int = 800, chunk_overlap: int = 100) -> list:
    """Protect fenced code blocks from being split, then run the normal
    recursive splitter, then restore the code blocks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = []
    for doc in docs:
        text = doc.page_content
        code_blocks = re.findall(r"```.*?```", text, flags=re.DOTALL)
        placeholder_text = text
        for i, block in enumerate(code_blocks):
            placeholder_text = placeholder_text.replace(block, f"__CODE_BLOCK_{i}__", 1)

        for raw_chunk in splitter.split_text(placeholder_text):
            restored = raw_chunk
            for i, block in enumerate(code_blocks):
                restored = restored.replace(f"__CODE_BLOCK_{i}__", block)
            if restored.strip():
                chunks.append(Document(page_content=restored, metadata=doc.metadata))
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", help="Folder of local doc files")
    parser.add_argument("--urls", nargs="*", help="One or more URLs to fetch and ingest")
    parser.add_argument("--persist_dir", default="./chroma_db")
    parser.add_argument("--chunk_size", type=int, default=800)
    parser.add_argument("--chunk_overlap", type=int, default=100)
    args = parser.parse_args()

    if not args.source_dir and not args.urls:
        parser.error("Pass --source_dir and/or --urls.")

    docs = []
    if args.source_dir:
        docs += load_local_files(args.source_dir)
    if args.urls:
        docs += load_urls(args.urls)
    print(f"Loaded {len(docs)} raw documents")
    if not docs:
        parser.error("No documents found -- check the path/URLs and file extensions.")

    chunks = code_aware_split(docs, args.chunk_size, args.chunk_overlap)
    print(f"Split into {len(chunks)} chunks")

    persist_path = Path(args.persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    with open(persist_path / "chunks.jsonl", "w") as f:
        for chunk in chunks:
            f.write(json.dumps({"page_content": chunk.page_content, "metadata": chunk.metadata}) + "\n")

    print(f"Embedding with {EMBEDDING_MODEL} (runs on CPU, first call downloads the model)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(chunks, embeddings, persist_directory=args.persist_dir)
    print(f"Indexed {len(chunks)} chunks into {args.persist_dir}")


if __name__ == "__main__":
    main()
