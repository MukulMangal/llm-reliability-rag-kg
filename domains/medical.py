# domains/medical.py
# Medical domain corpus — uses MedQuAD-style QA and medical Wikipedia passages

from datasets import load_dataset
import os
import json

CORPUS_PATH = "data/medical_corpus.json"
QA_PATH = "data/medical_qa.json"

# Seed medical facts for corpus
MEDICAL_FACTS = [
    "Diabetes mellitus is a metabolic disease characterized by high blood sugar. Symptoms include frequent urination, increased thirst, and increased hunger.",
    "Hypertension, or high blood pressure, is a condition where blood pressure is persistently elevated above 130/80 mmHg.",
    "Asthma is a chronic inflammatory disease of the airways causing wheezing, shortness of breath, chest tightness, and coughing.",
    "COVID-19 is caused by the SARS-CoV-2 coronavirus. Common symptoms include fever, cough, fatigue, and loss of taste or smell.",
    "Alzheimer's disease is a progressive neurological disorder that leads to memory loss and cognitive decline.",
    "The human heart beats approximately 60 to 100 times per minute in a healthy adult at rest.",
    "Insulin is a hormone produced by the pancreas that regulates blood glucose levels.",
    "Antibiotics are medications that kill or inhibit the growth of bacteria. They are ineffective against viral infections.",
    "The immune system consists of innate and adaptive immunity. White blood cells are key components.",
    "Chemotherapy uses drugs to destroy cancer cells. Common side effects include nausea, hair loss, and fatigue.",
    "The normal human body temperature is approximately 37°C (98.6°F).",
    "Anemia is a condition where the blood lacks enough healthy red blood cells to carry adequate oxygen.",
    "Depression is a mental health disorder characterized by persistent sadness, loss of interest, and low energy.",
    "Vaccines work by training the immune system to recognize and fight specific pathogens.",
    "The human genome contains approximately 3 billion base pairs and about 20,000 protein-coding genes.",
    "Stroke occurs when blood supply to part of the brain is cut off. FAST stands for Face, Arms, Speech, Time.",
    "Tuberculosis (TB) is caused by Mycobacterium tuberculosis. It primarily affects the lungs.",
    "Malaria is caused by Plasmodium parasites transmitted through the bites of infected mosquitoes.",
    "HIV attacks the immune system, specifically CD4 T cells, and can lead to AIDS if untreated.",
    "The liver performs over 500 functions including detoxification, protein synthesis, and bile production.",
]


def load_corpus() -> tuple:
    """Load medical domain corpus."""
    corpus_docs = list(MEDICAL_FACTS)
    qa_pairs = []

    # Try loading medical QA dataset
    try:
        print("Loading medical QA dataset...")
        med_qa = load_dataset("medalpaca/medical_meadow_medqa", split="train[:500]")
        for item in med_qa:
            q = item.get("input", "").strip()
            a = item.get("output", "").strip()
            if q and a and len(a) < 300:
                qa_pairs.append({"question": q, "answer": a, "all_answers": [a]})
                corpus_docs.append(f"Medical Q: {q} A: {a}")
        print(f"Medical QA loaded: {len(qa_pairs)} pairs")
    except Exception as e:
        print(f"Could not load medical QA dataset: {e}")
        print("Using seed medical facts only.")
        qa_pairs = [
            {"question": "What are the symptoms of diabetes?", "answer": "frequent urination, increased thirst, and increased hunger", "all_answers": ["frequent urination", "thirst", "hunger"]},
            {"question": "What causes hypertension?", "answer": "Hypertension can be caused by genetics, diet, stress, and lifestyle factors", "all_answers": ["genetics", "diet", "stress"]},
            {"question": "How do vaccines work?", "answer": "Vaccines train the immune system to recognize and fight specific pathogens", "all_answers": ["train the immune system"]},
        ]

    os.makedirs("data", exist_ok=True)
    with open(CORPUS_PATH, "w") as f:
        json.dump(corpus_docs, f)
    with open(QA_PATH, "w") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"Medical corpus ready: {len(corpus_docs)} docs")
    return corpus_docs, qa_pairs