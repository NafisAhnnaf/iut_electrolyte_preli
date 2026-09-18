# GridWise LLM — fallback execution image (BUP CSE Fest 2026 preliminary)
# Build:  docker build -t <registry>/gridwise-llm-preli:v1 .
# Run:    docker run -p 8000:8000 -e LLM_API_KEY=<your key> <registry>/gridwise-llm-preli:v1
# No secrets are baked in: the key is supplied at runtime via -e LLM_API_KEY.

FROM python:3.11-slim

WORKDIR /srv

# Install dependencies first so code edits don't bust the pip cache layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY sample_cases.json .

ENV PORT=8000
EXPOSE 8000

# Bind 0.0.0.0 so the service is reachable from outside the container.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
