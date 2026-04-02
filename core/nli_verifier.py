# core/nli_verifier.py
# NLI (Natural Language Inference) based hallucination verifier
# Uses cross-encoder/nli-deberta-v3-small to check if context ENTAILS the answer

from transformers import pipeline
import numpy as np

# Load NLI pipeline once at module level
print("Loading NLI model (first run may take a minute)...")
nli_pipeline = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-small",
    device=-1  # CPU; change to 0 for GPU
)


def nli_check(premise: str, hypothesis: str) -> dict:
    """
    Check if premise (context) entails hypothesis (claim).
    Returns entailment score between 0 and 1.
    """
    result = nli_pipeline(f"{premise} [SEP] {hypothesis}", truncation=True, max_length=512)
    label = result[0]["label"].upper()
    score = result[0]["score"]

    if label == "ENTAILMENT":
        return {"entailment": score, "contradiction": 1 - score, "verdict": "SUPPORTED"}
    elif label == "CONTRADICTION":
        return {"entailment": 1 - score, "contradiction": score, "verdict": "CONTRADICTED"}
    else:
        return {"entailment": 0.5, "contradiction": 0.5, "verdict": "NEUTRAL"}


def verify_answer_with_nli(answer: str, retrieved_docs: list) -> dict:
    """
    Verify an LLM answer against retrieved context using NLI.
    Checks each sentence in the answer against all retrieved docs.
    """
    if not retrieved_docs:
        return {
            "nli_score": 0.5,
            "verdict": "NO_CONTEXT",
            "details": []
        }

    # Combine all retrieved docs into one premise
    combined_context = " ".join([doc["document"] for doc in retrieved_docs[:3]])

    # Split answer into sentences for granular checking
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 20]

    if not sentences:
        return {
            "nli_score": 0.5,
            "verdict": "TOO_SHORT",
            "details": []
        }

    details = []
    entailment_scores = []

    for sentence in sentences[:5]:  # Check up to 5 sentences
        result = nli_check(combined_context, sentence)
        details.append({
            "sentence": sentence,
            "verdict": result["verdict"],
            "entailment_score": round(result["entailment"], 3)
        })
        entailment_scores.append(result["entailment"])

    avg_entailment = float(np.mean(entailment_scores))
    contradicted = sum(1 for d in details if d["verdict"] == "CONTRADICTED")

    if avg_entailment >= 0.7:
        overall_verdict = "SUPPORTED"
    elif avg_entailment >= 0.4:
        overall_verdict = "PARTIALLY_SUPPORTED"
    else:
        overall_verdict = "HALLUCINATED"

    return {
        "nli_score": round(avg_entailment, 3),
        "verdict": overall_verdict,
        "contradicted_sentences": contradicted,
        "total_sentences_checked": len(details),
        "details": details
    }