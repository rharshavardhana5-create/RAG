"""
cli.py

Interactive Q&A loop against your indexed documentation.

Usage:
    python cli.py --persist_dir ./chroma_db
"""

import argparse

from rag_pipeline import build_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist_dir", default="./chroma_db")
    parser.add_argument("--model", default="claude-sonnet-5")
    args = parser.parse_args()

    print("Loading retriever and model (first run downloads the embedding/reranker models)...")
    chain, _ = build_pipeline(args.persist_dir, args.model)

    print("Ready. Ask a question about your docs (Ctrl+C to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not question:
            continue
        answer = chain.invoke(question)
        print(f"\n{answer}")


if __name__ == "__main__":
    main()
