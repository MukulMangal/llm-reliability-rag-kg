# core/generator.py
# Groq LLM — RAG and Vanilla answer generation

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"


def generate_with_rag(query: str, retrieved_docs: list) -> dict:
    """Generate a grounded answer using retrieved context (RAG)."""
    context = "\n".join([f"- {doc['document']}" for doc in retrieved_docs])

    system_prompt = """You are a factual AI assistant. Answer questions using ONLY the provided context.
If the context doesn't contain enough information, say so clearly.
Do NOT make up facts. Be concise and accurate."""

    user_prompt = f"""Context:
{context}

Question: {query}

Answer based only on the context above:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=512
    )

    answer = response.choices[0].message.content.strip()
    return {
        "answer": answer,
        "model": MODEL,
        "tokens_used": response.usage.total_tokens
    }


def generate_vanilla(query: str) -> dict:
    """Generate answer WITHOUT any context (baseline)."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer the question directly."},
            {"role": "user", "content": query}
        ],
        temperature=0.1,
        max_tokens=512
    )
    return {
        "answer": response.choices[0].message.content.strip(),
        "model": MODEL
    }