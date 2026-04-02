# domains/science.py
# Science domain corpus — physics, chemistry, biology, space

import os
import json

CORPUS_PATH = "data/science_corpus.json"
QA_PATH = "data/science_qa.json"

SCIENCE_FACTS = [
    "The speed of light in a vacuum is approximately 299,792 kilometers per second (c).",
    "Newton's first law states that an object at rest stays at rest unless acted upon by an external force.",
    "Einstein's theory of general relativity describes gravity as a curvature of spacetime caused by mass.",
    "DNA (deoxyribonucleic acid) carries genetic information. Its double helix structure was discovered by Watson and Crick in 1953.",
    "The periodic table organizes elements by atomic number. It was created by Dmitri Mendeleev in 1869.",
    "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen.",
    "The Big Bang theory proposes the universe began as a hot dense point approximately 13.8 billion years ago.",
    "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape.",
    "The human body is made up of approximately 37 trillion cells.",
    "Quantum mechanics describes the behavior of matter and energy at atomic and subatomic scales.",
    "The Earth is approximately 4.5 billion years old and orbits the Sun at an average distance of 150 million km.",
    "The Milky Way galaxy contains between 100 and 400 billion stars.",
    "Newton's law of universal gravitation states every mass attracts every other mass.",
    "The water molecule consists of two hydrogen atoms bonded to one oxygen atom (H2O).",
    "CRISPR-Cas9 is a genome editing technology that allows scientists to modify DNA sequences.",
    "The Theory of Evolution by Natural Selection was proposed by Charles Darwin in 1859.",
    "Electrons orbit the nucleus of an atom. Protons are positively charged, neutrons are neutral.",
    "The Higgs boson, discovered in 2012, gives particles their mass.",
    "Mars has two moons: Phobos and Deimos. It takes 687 Earth days to orbit the Sun.",
    "The speed of sound in air at 20°C is approximately 343 meters per second.",
]

SCIENCE_QA = [
    {"question": "What is the speed of light?", "answer": "299,792 kilometers per second", "all_answers": ["299,792 km/s", "299792", "speed of light"]},
    {"question": "Who discovered the structure of DNA?", "answer": "Watson and Crick", "all_answers": ["Watson", "Crick", "Watson and Crick"]},
    {"question": "What is photosynthesis?", "answer": "process by which plants convert sunlight into glucose", "all_answers": ["convert sunlight", "glucose", "oxygen"]},
    {"question": "How old is the universe?", "answer": "approximately 13.8 billion years", "all_answers": ["13.8 billion", "13.8"]},
    {"question": "What is the atomic number of carbon?", "answer": "6", "all_answers": ["6"]},
    {"question": "What did Einstein's theory of relativity describe?", "answer": "gravity as a curvature of spacetime", "all_answers": ["spacetime", "gravity", "curvature"]},
    {"question": "When was the periodic table created?", "answer": "1869 by Dmitri Mendeleev", "all_answers": ["1869", "Mendeleev"]},
    {"question": "What is CRISPR?", "answer": "a genome editing technology", "all_answers": ["genome editing", "DNA modification"]},
]


def load_corpus() -> tuple:
    """Load science domain corpus."""
    corpus_docs = list(SCIENCE_FACTS)
    qa_pairs = list(SCIENCE_QA)

    os.makedirs("data", exist_ok=True)
    with open(CORPUS_PATH, "w") as f:
        json.dump(corpus_docs, f)
    with open(QA_PATH, "w") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"Science corpus ready: {len(corpus_docs)} docs, {len(qa_pairs)} QA pairs")
    return corpus_docs, qa_pairs