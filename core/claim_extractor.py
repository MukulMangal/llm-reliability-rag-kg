# core/claim_extractor.py
# spaCy-based SVO claim extraction

import spacy

nlp = spacy.load("en_core_web_sm")


def extract_claims(text: str) -> tuple:
    """Extract subject-predicate-object claims and named entities from text."""
    doc = nlp(text)
    claims = []

    for sent in doc.sents:
        sd = nlp(sent.text)

        for token in sd:
            # Pattern 1: Standard SVO
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subjects = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
                objects = [w for w in token.rights if w.dep_ in ("dobj", "attr", "pobj", "acomp")]
                for subj in subjects:
                    for obj in objects:
                        subj_span = " ".join([t.text for t in subj.subtree if not t.is_punct])
                        obj_span = " ".join([t.text for t in obj.subtree if not t.is_punct])
                        claims.append({
                            "subject": subj_span.strip(),
                            "predicate": token.lemma_,
                            "object": obj_span.strip(),
                            "sentence": sent.text.strip()
                        })

            # Pattern 2: "X was born in Y"
            if token.lemma_ == "bear" and token.dep_ in ("ROOT", "advcl", "relcl", "conj"):
                subjects = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
                for child in token.rights:
                    if child.dep_ == "prep" and child.lemma_ == "in":
                        for pobj in [w for w in child.rights if w.dep_ == "pobj"]:
                            for subj in subjects:
                                subj_span = " ".join([t.text for t in subj.subtree if not t.is_punct])
                                obj_span = " ".join([t.text for t in pobj.subtree if not t.is_punct])
                                claims.append({
                                    "subject": subj_span.strip(),
                                    "predicate": "born",
                                    "object": obj_span.strip(),
                                    "sentence": sent.text.strip()
                                })

            # Pattern 3: "X won/received Y"
            if token.lemma_ in ("win", "receive", "award") and token.pos_ == "VERB":
                subjects = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
                objects = [w for w in token.rights if w.dep_ in ("dobj", "nsubjpass")]
                for subj in subjects:
                    for obj in objects:
                        subj_span = " ".join([t.text for t in subj.subtree if not t.is_punct])
                        obj_span = " ".join([t.text for t in obj.subtree if not t.is_punct])
                        if subj_span.lower() in ("he", "she", "they", "it"):
                            persons = [e.text for e in doc.ents if e.label_ == "PERSON"]
                            if persons:
                                subj_span = persons[0]
                        claims.append({
                            "subject": subj_span.strip(),
                            "predicate": token.lemma_,
                            "object": obj_span.strip(),
                            "sentence": sent.text.strip()
                        })

    # Deduplicate
    seen, unique_claims = set(), []
    for c in claims:
        key = (c["subject"], c["predicate"], c["object"])
        if key not in seen:
            seen.add(key)
            unique_claims.append(c)

    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return unique_claims, entities