from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder
from langgraph.graph import END, START, StateGraph

from src.data_ingestion_pipeline.vector_store import QdrantVectorStore
from src.orange_agent.state import SupportState

load_dotenv()


@lru_cache(maxsize=1)
def _services() -> dict[str, Any]:
    """Load expensive models and infrastructure clients once per worker."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is missing from backend/.env")

    vector_store = QdrantVectorStore(collection_name="orange_telecom_store")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0,
    )
    return {"vector_store": vector_store, "reranker": reranker, "llm": llm}


def _latest_human_message(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage) or getattr(message, "type", None) == "human":
            return str(message.content)
    raise ValueError("The conversation does not contain a user message.")


def rewrite_query(state: SupportState) -> dict[str, str]:
    """Resolve short follow-up questions using the persisted conversation."""
    messages = state.get("messages", [])
    latest_question = _latest_human_message(messages)
    previous_turns = messages[:-1]

    if not previous_turns:
        return {"search_query": latest_question}

    llm = _services()["llm"]
    history = "\n".join(
        f"{getattr(message, 'type', 'assistant')}: {message.content}"
        for message in previous_turns[-8:]
    )
    prompt = [
        SystemMessage(content=(
            "You rewrite customer-support questions for document retrieval. "
            "Use the conversation history to make the latest question standalone. "
            "Do not answer it. Return only the rewritten search query."
        )),
        HumanMessage(content=(
            f"Conversation history:\n{history}\n\n"
            f"Latest question: {latest_question}\nStandalone query:"
        )),
    ]
    rewritten = llm.invoke(prompt).content
    return {"search_query": str(rewritten).strip() or latest_question}


def retrieve_documents(state: SupportState) -> dict[str, Any]:
    results = _services()["vector_store"].query(state["search_query"], top_k=8)
    return {"retrieved_chunks": results}


def validate_relevance(state: SupportState) -> dict[str, Any]:
    """Keep the highest-scoring passages that pass the relevance guardrail."""
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {"verified_chunks": []}

    query = state["search_query"]
    pairs = [
        [query, (chunk.get("metadata") or {}).get("text", "")]
        for chunk in chunks
    ]

    scores = _services()["reranker"].predict(pairs)
    threshold = float(os.getenv("RERANKER_THRESHOLD", "0.35"))

    verified = []

    for chunk, score in zip(chunks, scores):
        metadata = chunk.get("metadata") or {}

        if float(score) >= threshold and metadata.get("text"):
            verified.append({
                **metadata,
                "relevance_score": float(score),
            })

    verified.sort(
        key=lambda chunk: chunk["relevance_score"],
        reverse=True,
    )

    return {"verified_chunks": verified[:3]}


def _source_label(chunks: list[dict[str, Any]]) -> str:
    sources = []
    for chunk in chunks:
        title = chunk.get("title", "Support document")
        url = chunk.get("url", "")
        label = f"{title} ({url})" if url else title
        if label not in sources:
            sources.append(label)
    return "\n".join(f"- {source}" for source in sources)


def generate_answer(state: SupportState) -> dict[str, Any]:
    chunks = state["verified_chunks"]
    context = "\n\n---\n\n".join(chunk["text"] for chunk in chunks)
    history = "\n".join(
        f"{getattr(message, 'type', 'assistant')}: {message.content}"
        for message in state.get("messages", [])[-8:-1]
    )
    prompt = [
        SystemMessage(content=(
            "Tu es un assistant du service client Orange Burkina. "
            "Réponds en français, clairement et utilement. Utilise uniquement "
            "les informations vérifiées du contexte fourni. N'invente aucun prix, "
            "code USSD, délai, offre ou procédure. Si le contexte ne permet pas "
            "de répondre précisément, dis-le et recommande de contacter un conseiller."
        )),
        HumanMessage(content=(
            f"Historique récent:\n{history or 'Aucun'}\n\n"
            f"Contexte documentaire vérifié:\n{context}\n\n"
            f"Question: {_latest_human_message(state.get('messages', []))}\n\n"
            f"Sources disponibles:\n{_source_label(chunks)}\n\n"
            "Réponds à la question et cite uniquement les sources pertinentes ci-dessus."
        )),
    ]
    
    print(
        f"[generate_answer] Starting Gemini call. "
        f"Verified chunks: {len(chunks)}, context characters: {len(context)}",
        flush=True,
    )

    response = _services()["llm"].invoke(prompt)

    print("[generate_answer] Gemini call completed.", flush=True)

    answer = response.content
    return {"messages": [AIMessage(content=str(answer))]}


def safe_fallback(state: SupportState) -> dict[str, Any]:
    message = (
        "Je suis désolé, mais je n'ai pas trouvé d'information suffisamment "
        "fiable dans la documentation Orange disponible pour répondre à cette "
        "question. Pour éviter de vous induire en erreur, veuillez contacter "
        "un conseiller Orange."
    )
    return {"messages": [AIMessage(content=message)]}


def route_after_validation(state: SupportState) -> str:
    return "answer" if state.get("verified_chunks") else "fallback"


builder = StateGraph(SupportState)
builder.add_node("rewrite_query", rewrite_query)
builder.add_node("retrieve_documents", retrieve_documents)
builder.add_node("validate_relevance", validate_relevance)
builder.add_node("generate_answer", generate_answer)
builder.add_node("safe_fallback", safe_fallback)

builder.add_edge(START, "rewrite_query")
builder.add_edge("rewrite_query", "retrieve_documents")
builder.add_edge("retrieve_documents", "validate_relevance")
builder.add_conditional_edges(
    "validate_relevance",
    route_after_validation,
    {"answer": "generate_answer", "fallback": "safe_fallback"},
)
builder.add_edge("generate_answer", END)
builder.add_edge("safe_fallback", END)

# Aegra hosts this compiled LangGraph and persists thread state in PostgreSQL.
graph = builder.compile()
