# setup.py
# One-time setup: loads corpora and builds FAISS indexes for all domains

import os
import json
from dotenv import load_dotenv

load_dotenv()

from domains import general, medical, science
from core.retriever import build_index

DOMAINS = {
    "general": general,
    "medical": medical,
    "science": science,
}


def setup_domain(name: str, domain_module):
    print(f"\n{'='*50}")
    print(f"Setting up domain: {name.upper()}")
    print(f"{'='*50}")

    corpus_docs, qa_pairs = domain_module.load_corpus()

    index_path = f"embeddings/{name}_index.bin"
    corpus_path = f"embeddings/{name}_corpus.pkl"

    os.makedirs("embeddings", exist_ok=True)
    build_index(corpus_docs, index_path, corpus_path)

    print(f"✅ {name} domain ready!")
    return len(corpus_docs)


if __name__ == "__main__":
    print("🚀 LLM Reliability RAG+KG Setup")
    print("This will download datasets and build FAISS indexes.")
    print("General domain may take 10-15 minutes (Wikipedia download).\n")

    choice = input("Setup which domains? [all / general / medical / science]: ").strip().lower()

    if choice == "all":
        selected = list(DOMAINS.keys())
    elif choice in DOMAINS:
        selected = [choice]
    else:
        print("Invalid choice. Defaulting to 'science' (fastest).")
        selected = ["science"]

    for name in selected:
        setup_domain(name, DOMAINS[name])

    print("\n✅ All selected domains are ready!")
    print("Run the app with: streamlit run app/dashboard.py")