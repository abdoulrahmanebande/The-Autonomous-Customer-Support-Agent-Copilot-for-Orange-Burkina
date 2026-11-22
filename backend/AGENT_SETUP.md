# Run the Orange Burkina LangGraph agent with Aegra

This adds an Aegra-hosted LangGraph workflow while reusing the existing Qdrant
collection, multilingual embeddings, cross-encoder relevance gate, and Gemini
answer generation.

## Requirements

- Python 3.12+
- Docker running (Aegra uses it to start PostgreSQL in development)
- The existing Qdrant service running at `http://localhost:6333`
- The existing `orange_telecom_store` Qdrant collection populated
- A Google Gemini API key

## Install

Run these commands from the `backend/` directory:

```bash
python -m venv .venv
# Activate the virtual environment before continuing.
pip install --upgrade pip
pip install -r requirements-agent.txt
pip install aegra-cli
cp .env.example .env
```

Add your `GOOGLE_API_KEY` to `.env`. Keep this file private and never commit it.

## Start Aegra

From the same `backend/` directory:

```bash
aegra dev
```

Aegra reads `aegra.json`, loads the `orange_support` graph, starts its
development PostgreSQL service, and serves the Agent Protocol API at
`http://localhost:2026`.

Check health at `http://localhost:2026/health` and open the API docs at
`http://localhost:2026/docs`.

## What the graph does

1. Rewrites follow-up questions using the persisted conversation history.
2. Retrieves up to eight candidate passages from the existing Qdrant collection.
3. Applies the BAAI/bge-reranker-v2-m3 relevance threshold.
4. Generates a French answer using Gemini only from verified passages.
5. Returns a safe fallback when no passage passes the relevance gate.

Aegra/LangGraph persists each conversation thread in PostgreSQL. The graph
does not replace the original FastAPI endpoints or Flutter frontend; this is
an additional agent-serving path for testing and future integration.

## Important

- This branch is an integration scaffold and has not yet been run against your
  local Qdrant instance or Gemini key.
- The first run may download the embedding and reranker models.
- Aegra requires Python 3.12+, whereas the original README advertises Python
  3.10+. Use a separate Python 3.12 environment for this agent path.
- The graph uses the existing Qdrant collection. It does not automatically
  ingest documents into a new collection.
