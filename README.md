# The Autonomous Customer Support Agent Copilot for Orange Burkina 🛰️

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)](https://www.docker.com/)
[![Qdrant](https://img.shields.io/badge/qdrant-vector--db-green.svg)](https://qdrant.tech/)
[![Gemini API](https://img.shields.io/badge/gemini--api-llm--engine-orange.svg)](https://ai.google.dev/)
[![FastAPI](https://img.shields.io/badge/fastapi-backend-teal.svg)](https://fastapi.tiangolo.com/)

<br />
<div align="center">
  <a href="https://github.com/abdoulrahmanebande/The-Autonomous-Customer-Support-Agent-Copilot-for-Orange-Burkina">
    <img src="https://raw.githubusercontent.com/abdoulrahmanebande/The-Autonomous-Customer-Support-Agent-Copilot-for-Orange-Burkina/main/docs/banner.png" alt="Logo" width="100%" height="auto" onerror="this.src='https://placehold.co/800x400?text=Orange+Burkina+Copilot+Architecture'">
  </a>

  <h3 align="center">Production-Grade Conversational RAG Engine</h3>

  <p align="center">
    An advanced, secure Retrieval-Augmented Generation (RAG) backend designed to process complex French telecom documentation, operational workflows, and plans natively.
    <br />
    <a href="#system-architecture"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="#preview">View Demo</a>
    ·
    <a href="https://github.com/abdoulrahmanebande/The-Autonomous-Customer-Support-Agent-Copilot-for-Orange-Burkina/issues">Report Bug</a>
  </p>
</div>

---

## 📺 Project Preview <a name="preview"></a>

Below is a demonstration of the application in action, highlighting the administrative multi-file ingestion pipeline and real-time streaming conversational interface.

<div align="center">
  <img src="https://raw.githubusercontent.com/abdoulrahmanebande/The-Autonomous-Customer-Support-Agent-Copilot-for-Orange-Burkina/main/docs/preview.gif" alt="Project Demo" width="80%" onerror="this.style.display='none'">
  <p><i>Live demonstration of multi-document ingest and secure RAG pipeline</i></p>
</div>

---

## 🚀 Overview

This project delivers an enterprise-ready customer support agent architecture tailored for **Orange Burkina**. Designed to read, understand, and securely reason over internal data sheets, it solves critical LLM production challenges through advanced software patterns:
- **Zero Hallucination Risk:** Guarded by local asymmetric Cross-Encoder validation vectors.
- **Contextual Memory:** Dynamically rewrites user queries using historical chat state.
- **Asynchronous Scalability:** Decoupled multi-file upload execution via background thread pools.

---

## 💼 Business Value & Operational Strategy

- **Risk Mitigation:** The local validation gate blocks irrelevant or adversarial user queries, protecting the enterprise from legal or brand liability.
- **Token Cost Containment:** Broad, cheap local filtering prevents bloated document sizes from hitting external generative APIs, dramatically slicing operational compute costs.
- **High Operational Availability:** Administrative ingest functions run concurrently with customer queries, allowing hot-swaps of internal policy documents without introducing server downtime.

---

## 🏗 System Architecture <a name="system-architecture"></a>

```text
                  [ USER CONVERSATIONAL INPUT ]
                                │
                                ▼
            [ Contextual History Query Rewriter ]
              (Injects Chat Memory via Gemini)
                                │
                                ▼
               [ Qdrant Custom Vector Search ]
               (Fetches Top-8 Candidates: k=8)
                                │
                                ▼
         [ GUARDRAIL LAYER: BAAI/bge-reranker-v2-m3 ]
        (Asymmetric Cross-Encoder Relevance Validation)
                                │
             🥇 Is Top Score >= Threshold (0.45)?
             ├── YES ──► Pass Top-3 Clean Chunks to Gemini Context
             └── NO  ──► [DETERMINISTIC FALLBACK TRIGGERED]
                                │
                                ▼
       [ Safe Final Synthesis / Streaming Markdown Output ] 
```