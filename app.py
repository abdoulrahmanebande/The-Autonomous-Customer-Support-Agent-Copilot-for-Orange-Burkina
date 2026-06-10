from src.data_ingestion_pipeline.data_loader import load_all_documents
from src.retrieval_pipeline.rag_search import RAGSearch

if __name__ == '__main__':
  rag_search = RAGSearch()
  
  summary = rag_search.search_and_summarize(query='Qui est Abdoul-Rahmane BANDE ?',top_k=3)
  print(f"Here is the summary:\n{summary}")