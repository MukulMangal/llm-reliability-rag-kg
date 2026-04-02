# 🧠 LLM Reliability with RAG + Knowledge Graph + NLI

A system that improves LLM reliability by combining RAG, Knowledge Graph verification (Wikidata), and NLI (DeBERTa) to detect hallucinations in real time.

## Quick Start
1. Clone the repo
2. Run `pip install -r requirements.txt`
3. Run `python setup.py`
4. Run `streamlit run app/dashboard.py`

## Tech Stack
- Groq (Llama 3.1 8B)
- FAISS + Sentence-BERT
- Wikidata SPARQL
- DeBERTa NLI
- Streamlit

## 📊 Benchmark Results (Science Domain)

| Metric | RAG | Vanilla LLM |
|--------|-----|-------------|
| Exact Match | **87.5%** | 87.5% |
| ROUGE-L | **0.462** | 0.150 |
| Avg NLI Score | **0.387** | — |

> RAG answers are 3x more faithful to facts than Vanilla LLM (ROUGE-L 0.46 vs 0.15).
> Vanilla LLM hallucinates extra details not grounded in context.