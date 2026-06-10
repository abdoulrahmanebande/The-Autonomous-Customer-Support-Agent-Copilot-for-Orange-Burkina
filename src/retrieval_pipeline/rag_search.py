import os
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage
from qdrant_client import QdrantClient
from src.data_ingestion_pipeline.vector_store import QdrantVectorStore 
from src.logger import logging
from dotenv import load_dotenv
load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

class Payload(BaseModel):
  url: str=Field(description='The URL of the content.')
  title: str=Field(description='The title of the content.')
  text: str=Field(description="The actual content's text.")
  
class RAGSearch:
  def __init__(
    self,
    collection_name: str='orange_telecom_store',
    llm_model: str='gemini-2.5-flash'
  ):
    self.collection_name = collection_name
    self.vector_store = QdrantVectorStore(collection_name=collection_name)
    self.client = QdrantClient(url='http://localhost:6333')
    
    collection_response = self.client.get_collections()
    existing_collections = [col.name for col in collection_response.collections]
    
    if self.collection_name not in existing_collections:
      from src.data_ingestion_pipeline.data_loader import load_all_documents
      documents = load_all_documents("data")
      self.vector_store.build_from_documents(documents)
      
    llm = ChatGoogleGenerativeAI(model=llm_model)
    self.llm_with_structured = llm.with_structured_output(Payload)
    logging.info(f"Gemini LLM initialized: {llm_model}")
    
  def search_and_summarize(self, query: str, top_k: int=5):
    results = self.vector_store.query(query, top_k=top_k)
    
    texts = []
    source_urls = []
    source_titles = []
    
    for r in results:
        if r.get('metadata'):
            texts.append(r['metadata'].get('text', ''))
            source_urls.append(r['metadata'].get('url', ''))
            source_titles.append(r['metadata'].get('title', ''))
            
    context = "\n---\n".join(texts)
    
    if not context.strip():
      return 'I could not find relevant context in my Database to answer this question.'
    
    messages = [
      SystemMessage(content=(
          "You are a helpful customer support agent. Answer the user's question based strictly on the context. "
          "Provide your complete answer in the 'text' field of the schema."
        )
      ),
      HumanMessage(content=f"Content:\n{context}\n\nQuestion:{query}\n\nAnswer:")
    ]
    
    response = self.llm_with_structured.invoke(messages)
    
    response.url = source_urls[0] if source_urls else "No URL link found"
    response.title = source_titles[0] if source_titles else "No Title found"
    
    return response
    
