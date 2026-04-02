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