FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 7860

ENV GROQ_API_KEY=""

CMD ["streamlit", "run", "app/dashboard.py", "--server.port=7860", "--server.address=0.0.0.0"]