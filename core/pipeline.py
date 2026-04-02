# core/pipeline.py
# Full pipeline combining retrieval, generation, KG + NLI verification

import time
from core.retriever import retrieve_top_k
from core.generator import generate_with_rag, generate_vanilla
from core.claim_extractor import extract_claims
from core.kg_verifier import verify_claim_wikidata
from core.nli_verifier import verify_answer_with_nli


def validate_with_kg(llm_response: str) -> dict:
    """Validate LLM response claims against Wikidata KG."""
    claims, entities = extract_claims(llm_response)
    validation_results = []
    verified_count = contradicted_count = 0

    for claim in claims[:5]:
        result = verify_claim_wikidata(claim["subject"], claim["predicate"], claim["object"])
        result["claim"] = claim
        validation_results.append(result)
        if result["status"] == "VERIFIED":
            verified_count += 1
        elif result["status"] == "CONTRADICTED":
            contradicted_count += 1

    checkable = [r for r in validation_results if r["status"] in ("VERIFIED", "CONTRADICTED")]
    hallucination_score = contradicted_count / len(checkable) if checkable else 0.0

    return {
        "claims_checked": len(validation_results),
        "verified": verified_count,
        "contradicted": contradicted_count,
        "hallucination_score": round(hallucination_score, 3),
        "kg_confidence": round(1.0 - hallucination_score, 3),
        "entities_found": entities,
        "claim_results": validation_results
    }


def full_pipeline(query: str, index, corpus_docs: list, top_k: int = 5, use_nli: bool = True) -> dict:
    """
    Run the complete RAG + KG + NLI reliability pipeline.

    Steps:
      1. Retrieve top-k relevant documents
      2. Generate RAG answer (grounded)
      3. Generate Vanilla answer (baseline)
      4. Validate with KG (Wikidata SPARQL)
      5. Validate with NLI (DeBERTa)
      6. Compute combined reliability score
    """
    t0 = time.time()

    # Step 1: Retrieval
    retrieved_docs = retrieve_top_k(query, index, corpus_docs, k=top_k)
    t1 = time.time()

    # Step 2 & 3: Generation
    rag_result = generate_with_rag(query, retrieved_docs)
    t2 = time.time()
    vanilla_result = generate_vanilla(query)
    t3 = time.time()

    # Step 4: KG Verification
    kg_validation = validate_with_kg(rag_result["answer"])
    t4 = time.time()

    # Step 5: NLI Verification
    nli_result = {}
    if use_nli:
        nli_result = verify_answer_with_nli(rag_result["answer"], retrieved_docs)
    t5 = time.time()

    # Step 6: Combined reliability score
    kg_conf = kg_validation.get("kg_confidence", 0.5)
    nli_conf = nli_result.get("nli_score", 0.5) if use_nli else 0.5
    combined_reliability = round((kg_conf * 0.5) + (nli_conf * 0.5), 3)

    return {
        "query": query,
        "rag_answer": rag_result["answer"],
        "vanilla_answer": vanilla_result["answer"],
        "retrieved_docs": retrieved_docs,
        "kg_validation": kg_validation,
        "nli_validation": nli_result,
        "combined_reliability_score": combined_reliability,
        "timing": {
            "retrieval_ms": round((t1 - t0) * 1000, 1),
            "rag_generation_ms": round((t2 - t1) * 1000, 1),
            "vanilla_generation_ms": round((t3 - t2) * 1000, 1),
            "kg_validation_ms": round((t4 - t3) * 1000, 1),
            "nli_validation_ms": round((t5 - t4) * 1000, 1),
            "total_ms": round((t5 - t0) * 1000, 1)
        },
        "tokens_used": rag_result["tokens_used"]
    }