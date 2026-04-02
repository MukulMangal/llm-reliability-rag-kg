# evaluation/evaluator.py
# Evaluation pipeline: Exact Match, ROUGE-L, BERTScore, Hallucination Rate

import re
import json
import time
import os
from rouge_score import rouge_scorer

try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("BERTScore not available. Install with: pip install bert-score")

from core.retriever import retrieve_top_k
from core.generator import generate_with_rag, generate_vanilla
from core.nli_verifier import verify_answer_with_nli

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower().strip())


def exact_match(pred: str, gold_answers: list) -> bool:
    pred_norm = normalize(pred)
    return any(normalize(ans) in pred_norm for ans in gold_answers)


def rouge_l_score(pred: str, gold: str) -> float:
    scores = scorer.score(gold, pred)
    return round(scores["rougeL"].fmeasure, 4)


def run_evaluation(qa_pairs: list, index, corpus_docs: list,
                   n_samples: int = 50, domain: str = "general",
                   use_nli: bool = True) -> dict:
    """
    Run full evaluation on QA pairs.
    Compares RAG vs Vanilla on: Exact Match, ROUGE-L, NLI score.
    """
    eval_set = qa_pairs[:n_samples]
    results_log = []

    rag_em = vanilla_em = 0
    rag_rouge = vanilla_rouge = 0.0
    rag_nli_scores = []

    print(f"\nEvaluating {len(eval_set)} samples on domain: {domain}")
    print("=" * 50)

    for i, item in enumerate(eval_set):
        try:
            docs = retrieve_top_k(item["question"], index, corpus_docs, k=5)
            rag_ans = generate_with_rag(item["question"], docs)["answer"]
            van_ans = generate_vanilla(item["question"])["answer"]

            rag_hit = exact_match(rag_ans, item["all_answers"])
            van_hit = exact_match(van_ans, item["all_answers"])
            rag_r = rouge_l_score(rag_ans, item["answer"])
            van_r = rouge_l_score(van_ans, item["answer"])

            nli_result = {}
            if use_nli:
                nli_result = verify_answer_with_nli(rag_ans, docs)
                rag_nli_scores.append(nli_result.get("nli_score", 0.5))

            if rag_hit: rag_em += 1
            if van_hit: vanilla_em += 1
            rag_rouge += rag_r
            vanilla_rouge += van_r

            results_log.append({
                "question": item["question"],
                "gold": item["answer"],
                "rag_answer": rag_ans,
                "vanilla_answer": van_ans,
                "rag_exact_match": rag_hit,
                "vanilla_exact_match": van_hit,
                "rag_rouge_l": rag_r,
                "vanilla_rouge_l": van_r,
                "nli_score": nli_result.get("nli_score", None),
                "nli_verdict": nli_result.get("verdict", None),
            })

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(eval_set)} | RAG EM: {rag_em}/{i+1} | Vanilla EM: {vanilla_em}/{i+1}")

            time.sleep(0.3)

        except Exception as e:
            print(f"  Skipped item {i}: {e}")

    n = len(results_log)
    summary = {
        "domain": domain,
        "samples_evaluated": n,
        "rag_exact_match_pct": round(rag_em / n * 100, 2) if n else 0,
        "vanilla_exact_match_pct": round(vanilla_em / n * 100, 2) if n else 0,
        "rag_rouge_l": round(rag_rouge / n, 4) if n else 0,
        "vanilla_rouge_l": round(vanilla_rouge / n, 4) if n else 0,
        "rag_avg_nli_score": round(sum(rag_nli_scores) / len(rag_nli_scores), 3) if rag_nli_scores else None,
        "em_improvement_pct": round((rag_em - vanilla_em) / n * 100, 2) if n else 0,
        "results": results_log
    }

    os.makedirs("evaluation/results", exist_ok=True)
    out_path = f"evaluation/results/{domain}_eval.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"RESULTS — {domain.upper()}")
    print(f"{'='*50}")
    print(f"RAG Exact Match    : {summary['rag_exact_match_pct']}%")
    print(f"Vanilla EM         : {summary['vanilla_exact_match_pct']}%")
    print(f"Improvement        : +{summary['em_improvement_pct']}%")
    print(f"RAG ROUGE-L        : {summary['rag_rouge_l']}")
    print(f"Vanilla ROUGE-L    : {summary['vanilla_rouge_l']}")
    if summary["rag_avg_nli_score"]:
        print(f"RAG Avg NLI Score  : {summary['rag_avg_nli_score']}")
    print(f"Saved to: {out_path}")

    return summary