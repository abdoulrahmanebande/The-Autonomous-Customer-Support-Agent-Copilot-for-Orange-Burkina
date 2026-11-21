from __future__ import annotations

from typing import Annotated, TypedDict, Any
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class SupportState(TypedDict, total=False):
    # Aegra/LangGraph persists this message history per conversation thread.
    messages: Annotated[list[AnyMessage], add_messages]
    search_query: str
    retrieved_chunks: list[dict[str, Any]]
    verified_chunks: list[dict[str, Any]]
