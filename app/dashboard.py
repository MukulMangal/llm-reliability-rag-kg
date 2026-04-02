# app/dashboard.py
# Streamlit dashboard for LLM Reliability RAG + KG + NLI system

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.retriever import build_index, load_index, retrieve_top_k
from core.pipeline import full_pipeline

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Reliability Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    border: 1px solid #313244;
}
.reliability-high { color: #a6e3a1; font-size: 2rem; font-weight: bold; }
.reliability-mid  { color: #f9e2af; font-size: 2rem; font-weight: bold; }
.reliability-low  { color: #f38ba8; font-size: 2rem; font-weight: bold; }
.verdict-SUPPORTED     { color: #a6e3a1; font-weight: bold; }
.verdict-PARTIALLY_SUPPORTED { color: #f9e2af; font-weight: bold; }
.verdict-HALLUCINATED  { color: #f38ba8; font-weight: bold; }
.verdict-VERIFIED      { color: #a6e3a1; font-weight: bold; }
.verdict-CONTRADICTED  { color: #f38ba8; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    domain = st.selectbox("🌐 Domain", ["General", "Medical", "Science"])
    top_k = st.slider("📄 Retrieved Documents", 3, 10, 5)
    use_nli = st.toggle("🔬 Enable NLI Verification", value=True)
    use_kg = st.toggle("🕸️ Enable KG Verification", value=True)
    st.divider()
    st.caption("LLM Reliability RAG + KG + NLI System v2.0")


@st.cache_resource(show_spinner="Loading index...")
def get_index(domain_name: str):
    """Load or build FAISS index for selected domain."""
    domain_lower = domain_name.lower()
    index_path = f"embeddings/{domain_lower}_index.bin"
    corpus_path = f"embeddings/{domain_lower}_corpus.pkl"
    data_path = f"data/{domain_lower}_corpus.json"

    if os.path.exists(index_path) and os.path.exists(corpus_path):
        return load_index(index_path, corpus_path)

    if not os.path.exists(data_path):
        st.error(f"No corpus found for domain '{domain_name}'. Please run the setup script first.")
        st.stop()

    with open(data_path) as f:
        corpus_docs = json.load(f)

    return build_index(corpus_docs, index_path, corpus_path)


# ── Main UI ───────────────────────────────────────────────────
st.title("🧠 LLM Reliability Dashboard")
st.caption("RAG + Knowledge Graph + NLI — Hallucination Detection System")
st.divider()

# Load index
with st.spinner(f"Loading {domain} knowledge base..."):
    try:
        index, corpus_docs = get_index(domain)
        st.success(f"✅ {domain} knowledge base loaded ({len(corpus_docs):,} documents)")
    except Exception as e:
        st.error(f"Failed to load index: {e}")
        st.stop()

# ── Query Input ───────────────────────────────────────────────
st.subheader("🔍 Ask a Question")
query = st.text_input("Enter your question:", placeholder="e.g. Where was Einstein born?")
run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

if run_btn and query.strip():
    with st.spinner("Running RAG + KG + NLI pipeline..."):
        result = full_pipeline(query, index, corpus_docs, top_k=top_k, use_nli=use_nli)

    # ── Reliability Score Gauge ───────────────────────────────
    st.divider()
    reliability = result["combined_reliability_score"]
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=reliability * 100,
            title={"text": "Combined Reliability Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#a6e3a1" if reliability > 0.7 else "#f9e2af" if reliability > 0.4 else "#f38ba8"},
                "steps": [
                    {"range": [0, 40], "color": "#313244"},
                    {"range": [40, 70], "color": "#45475a"},
                    {"range": [70, 100], "color": "#585b70"},
                ],
            }
        ))
        fig.update_layout(height=250, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        kg = result["kg_validation"]
        st.metric("🕸️ KG Confidence", f"{kg['kg_confidence'] * 100:.0f}%")
        st.metric("✅ Claims Verified", f"{kg['verified']} / {kg['claims_checked']}")
        st.metric("❌ Contradicted", kg["contradicted"])

    with col3:
        if use_nli and result.get("nli_validation"):
            nli = result["nli_validation"]
            st.metric("🔬 NLI Score", f"{nli['nli_score'] * 100:.0f}%")
            verdict = nli.get("verdict", "N/A")
            color = "green" if verdict == "SUPPORTED" else "orange" if verdict == "PARTIALLY_SUPPORTED" else "red"
            st.markdown(f"**NLI Verdict:** :{color}[{verdict}]")
            st.metric("📝 Sentences Checked", nli.get("total_sentences_checked", 0))

    # ── Answer Comparison ─────────────────────────────────────
    st.divider()
    st.subheader("📊 Answer Comparison")
    col_rag, col_van = st.columns(2)

    with col_rag:
        st.markdown("### 🤖 RAG Answer")
        st.info(result["rag_answer"])

    with col_van:
        st.markdown("### 📉 Vanilla LLM (Baseline)")
        st.warning(result["vanilla_answer"])

    # ── Claim Verification Table ──────────────────────────────
    if kg["claim_results"]:
        st.divider()
        st.subheader("🕸️ Claim-by-Claim KG Verification")
        for r in kg["claim_results"]:
            c = r["claim"]
            status = r["status"]
            icon = "✅" if status == "VERIFIED" else "❌" if status == "CONTRADICTED" else "⚠️"
            with st.expander(f"{icon} [{c['subject']}] — {c['predicate']} → [{c['object']}]  |  {status}"):
                st.write(f"**Sentence:** {c['sentence']}")
                st.write(f"**Reason:** {r['reason']}")
                if r.get("wikidata_qid"):
                    st.write(f"**Wikidata QID:** [{r['wikidata_qid']}](https://www.wikidata.org/wiki/{r['wikidata_qid']})")

    # ── NLI Sentence Detail ───────────────────────────────────
    if use_nli and result.get("nli_validation", {}).get("details"):
        st.divider()
        st.subheader("🔬 NLI Sentence-Level Analysis")
        for d in result["nli_validation"]["details"]:
            icon = "✅" if d["verdict"] == "SUPPORTED" else "❌" if d["verdict"] == "CONTRADICTED" else "⚠️"
            st.markdown(f"{icon} **{d['verdict']}** (score: {d['entailment_score']}) — {d['sentence']}")

    # ── Retrieved Documents ───────────────────────────────────
    st.divider()
    with st.expander("📄 Retrieved Documents"):
        for i, doc in enumerate(result["retrieved_docs"]):
            st.markdown(f"**[{i+1}] Score: {doc['score']:.3f}**")
            st.caption(doc["document"][:300])

    # ── Timing ────────────────────────────────────────────────
    st.divider()
    st.subheader("⏱️ Pipeline Timing")
    timing = result["timing"]
    fig2 = px.bar(
        x=list(timing.keys()),
        y=list(timing.values()),
        labels={"x": "Step", "y": "Time (ms)"},
        color=list(timing.values()),
        color_continuous_scale="viridis"
    )
    fig2.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# ── Evaluation Results Tab ────────────────────────────────────
st.divider()
st.subheader("📈 Evaluation Results")

eval_files = [f for f in os.listdir("evaluation/results") if f.endswith("_eval.json")] if os.path.exists("evaluation/results") else []

if eval_files:
    selected_eval = st.selectbox("Select evaluation results:", eval_files)
    with open(f"evaluation/results/{selected_eval}") as f:
        eval_data = json.load(f)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RAG Exact Match", f"{eval_data['rag_exact_match_pct']}%")
    c2.metric("Vanilla Exact Match", f"{eval_data['vanilla_exact_match_pct']}%")
    c3.metric("RAG ROUGE-L", eval_data["rag_rouge_l"])
    c4.metric("EM Improvement", f"+{eval_data['em_improvement_pct']}%")

    fig3 = go.Figure(data=[
        go.Bar(name="RAG", x=["Exact Match %", "ROUGE-L"], y=[eval_data["rag_exact_match_pct"], eval_data["rag_rouge_l"] * 100]),
        go.Bar(name="Vanilla", x=["Exact Match %", "ROUGE-L"], y=[eval_data["vanilla_exact_match_pct"], eval_data["vanilla_rouge_l"] * 100]),
    ])
    fig3.update_layout(barmode="group", height=350, title="RAG vs Vanilla LLM Performance")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No evaluation results yet. Run `python run_evaluation.py` to generate them.")