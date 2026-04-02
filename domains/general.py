# domains/general.py
# General domain — TriviaQA + Wikipedia corpus loader

from datasets import load_dataset
import os
import json

CORPUS_PATH = "data/general_corpus.json"
QA_PATH = "data/general_qa.json"


def load_corpus(max_wiki: int = 10000, max_trivia: int = 2000) -> tuple:
    """Load general domain corpus from TriviaQA + Wikipedia."""
    corpus_docs = []
    qa_pairs = []

    print("Loading TriviaQA...")
    trivia = load_dataset("trivia_qa", "rc.nocontext", split=f"train[:{max_trivia}]")
    for item in trivia:
        question = item["question"]
        answers = item["answer"]["aliases"] if item["answer"]["aliases"] else [item["answer"]["value"]]
        answer = answers[0] if answers else ""
        if question and answer:
            qa_pairs.append({"question": question, "answer": answer, "all_answers": answers})
            corpus_docs.append(f"Q: {question} A: {answer}")

    print(f"TriviaQA loaded: {len(corpus_docs)} pairs")

    print(f"Loading {max_wiki} Wikipedia passages...")
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    wiki_count = 0
    for item in wiki:
        if wiki_count >= max_wiki:
            break
        title = item.get("title", "").strip()
        text = item.get("text", "").strip()
        if text and len(text) > 100:
            words = text.split()
            chunks = [words[j:j + 200] for j in range(0, min(len(words), 600), 200)]
            for chunk in chunks:
                corpus_docs.append(f"{title}: {' '.join(chunk)}")
                wiki_count += 1
                if wiki_count >= max_wiki:
                    break

    os.makedirs("data", exist_ok=True)
    with open(CORPUS_PATH, "w") as f:
        json.dump(corpus_docs, f)
    with open(QA_PATH, "w") as f:
        json.dump(qa_pairs[:500], f, indent=2)

    print(f"General corpus ready: {len(corpus_docs)} docs, {len(qa_pairs)} QA pairs")
    return corpus_docs, qa_pairs