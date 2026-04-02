# core/kg_verifier.py
# Wikidata SPARQL-based claim verification

from SPARQLWrapper import SPARQLWrapper, JSON
import time

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)
sparql.addCustomHttpHeader("User-Agent", "LLM-Reliability-RAG-KG/2.0 (research)")

PREDICATE_MAP = {
    "bear": "P19", "born": "P19",
    "win": "P166", "won": "P166", "receive": "P166",
    "die": "P20", "write": "P50",
    "discover": "P61", "invent": "P61",
    "locate": "P131", "found": "P571",
    "compose": "P86", "direct": "P57",
    "publish": "P123", "create": "P170",
}


def fuzzy_match(a: str, b: str) -> bool:
    stops = {"the", "a", "an", "of", "in", "for", "and", "or"}
    a_words = set(a.lower().split()) - stops
    b_words = set(b.lower().split()) - stops
    if not a_words or not b_words:
        return False
    return len(a_words & b_words) / min(len(a_words), len(b_words)) >= 0.5


def search_wikidata_entity(entity_name: str) -> str:
    query = f"""
    SELECT ?item WHERE {{
      ?item rdfs:label "{entity_name}"@en .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
    }}
    LIMIT 1
    """
    try:
        sparql.setQuery(query)
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]
        if bindings:
            return bindings[0]["item"]["value"].split("/")[-1]
    except Exception:
        pass
    return None


def verify_claim_wikidata(subject: str, predicate: str, obj: str) -> dict:
    if subject.strip().lower() in ("he", "she", "they", "it", "i", "we"):
        return {
            "verified": None, "confidence": 0.5, "status": "SKIPPED",
            "reason": "Pronoun subject — cannot resolve", "wikidata_qid": None
        }

    qid = search_wikidata_entity(subject)
    time.sleep(0.5)

    if not qid:
        return {
            "verified": None, "confidence": 0.0, "status": "UNVERIFIABLE",
            "reason": f'Entity "{subject}" not found in Wikidata', "wikidata_qid": None
        }

    prop = PREDICATE_MAP.get(predicate.lower())
    if prop:
        query = f"""
        SELECT ?valueLabel WHERE {{
          wd:{qid} wdt:{prop} ?value .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
        }}
        LIMIT 5
        """
        try:
            sparql.setQuery(query)
            results = sparql.query().convert()
            kg_values = [b["valueLabel"]["value"].lower() for b in results["results"]["bindings"]]
            time.sleep(0.3)
            obj_lower = obj.lower().strip()
            matched = any(
                obj_lower in v or v in obj_lower or fuzzy_match(obj_lower, v)
                for v in kg_values
            )
            return {
                "verified": matched,
                "confidence": 0.9 if matched else 0.15,
                "status": "VERIFIED" if matched else "CONTRADICTED",
                "reason": f"KG values: {kg_values}",
                "wikidata_qid": qid
            }
        except Exception:
            pass

    return {
        "verified": None, "confidence": 0.5, "status": "PARTIAL",
        "reason": f"Entity found (QID: {qid}) but predicate not mapped",
        "wikidata_qid": qid
    }