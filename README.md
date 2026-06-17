# 🛰️ The Autonomous Customer Support Agent Copilot for Orange Burkina

An advanced, asynchronous Retrieval-Augmented Generation (RAG) backend engine designed to analyze telecom documentation for Orange Burkina. This system dynamically processes French technical sheets, subscription criteria, and operational workflows natively. It integrates conversational memory, history rewriting, vector retrieval, cross-encoder guardrail re-ranking, and dynamic streaming multi-format file ingestion into a secure production architecture.

---

## 💼 Executive Value Proposition (For Management)

This solution directly addresses operational efficiency, cost reduction, and security compliance for Orange Burkina customer service:
* **Zero Hallucination Risk:** By incorporating a cross-encoder verification layer, the AI is structurally blocked from providing false information or fabrications to customers, safeguarding corporate reputation.
* **Cost & Performance Efficiency:** Instead of processing massive quantities of raw text data through cloud APIs—which incurs high token costs—the backend utilizes local, lightweight open-source models to filter and verify data on-site before calling final generation layers.
* **Uninterrupted Service Availability:** Administrators can continuously ingest new technical files or update subscription plans while customers are actively chatting. The upload pipeline is completely isolated using background thread pools, maintaining zero system downtime.

---

## 🏗️ Core Architectural Pipeline

The system is built as a robust pipeline prioritizing defensive structure and asynchronous background isolation:

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
