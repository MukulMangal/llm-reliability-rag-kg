# run_evaluation.py
# Run evaluation benchmarks for selected domain

import json
import os
from dotenv import load_dotenv

load_dotenv()

from core.retriever import load_index
from evaluation.evaluator import run_evaluation


def main():
    domain = input("Which domain to evaluate? [general / medical / science]: ").strip().lower()
    n = int(input("How many samples? (e.g. 50): ").strip())
    use_nli = input("Enable NLI verification? [y/n]: ").strip().lower() == "y"

    index_path = f"embeddings/{domain}_index.bin"
    corpus_path = f"embeddings/{domain}_corpus.pkl"
    qa_path = f"data/{domain}_qa.json"

    if not os.path.exists(index_path):
        print(f"No index found for domain '{domain}'. Run setup.py first.")
        return

    index, corpus_docs = load_index(index_path, corpus_path)

    with open(qa_path) as f:
        qa_pairs = json.load(f)

    results = run_evaluation(
        qa_pairs=qa_pairs,
        index=index,
        corpus_docs=corpus_docs,
        n_samples=n,
        domain=domain,
        use_nli=use_nli
    )

    print(f"\nDone! Results saved to evaluation/results/{domain}_eval.json")


if __name__ == "__main__":
    main()