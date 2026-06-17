import numpy as np
from sentence_transformers import SentenceTransformer
from src.data_ingestion_pipeline.data_loader import load_all_documents
from src.data_ingestion_pipeline.embedding import EmbeddingPipeline
from src.logger import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from typing import List, Any

class QdrantVectorStore:
  def __init__(
    self,
    collection_name :str='orange_telecom_store',
    embedding_model: str='intfloat/multilingual-e5-base'
  ):
    self.collection_name = collection_name
    self.embedding_model = embedding_model
    self.model = SentenceTransformer(embedding_model)
    
    # Connect to running docker Qdrant container
    self.client = QdrantClient(url='http://localhost:6333')
    
    logging.info(f"Loaded embedding model: {embedding_model}")
    logging.info(f"Connected to Qdrant server for container: {collection_name}")
    
  def build_from_documents(self, documents: List[Any]):
    logging.info(f"Building vector store from {len(documents)} raw docs...")
    
    emb_pipeline = EmbeddingPipeline()
    chunks = emb_pipeline.chunk_documents(documents)
    embeddings = emb_pipeline.embed_chunks(chunks)
    
    metadatas = [
      {
        'text': chunk.page_content,
        'title': chunk.metadata.get('title', 'Unknown Title'),
        'url': chunk.metadata.get('url', 'Unknown URL')
      }
      for chunk in chunks
    ]
    
    self.add_embedding(np.array(embeddings).astype('float32'), metadatas)
    
  def add_embedding(self, embeddings: np.ndarray, metadatas: List[Any]=None):
    dim = embeddings.shape[1]
    
    collection_response = self.client.get_collections()
    existing_collections = [col.name for col in collection_response.collections]
    
    if self.collection_name not in existing_collections:
      self.client.create_collection(
        collection_name=self.collection_name,
        vectors_config= VectorParams(size=dim, distance=Distance.COSINE)
      )
      logging.info(f"created new Qdrant collection: {self.collection_name} with dim {dim}")
    
    #Package embeddings and metadata into Qdrant Points
    points = []
    for idx, (vector, meta) in enumerate(zip(embeddings, metadatas)):
      points.append(
        PointStruct(
          id=idx,
          vector=vector.tolist(),
          payload=meta
        )
      )
    
    self.client.upsert(
      collection_name=self.collection_name,
      points=points,
      wait=True
    )
    logging.info(f"Added {embeddings.shape[0]} vectors to Qdrant collection")
    
  def search(self, query_embedding: np.ndarray, top_k: int=5):
    query_vector = query_embedding[0].tolist()
    
    search_results = self.client.query_points(
      collection_name=self.collection_name,
      query=query_vector,
      limit=top_k
    )
    
    results=[]
    for point in search_results.points:
      # Convert the internal Qdrant object directly to a clean Python dict
      hit = point.model_dump()
      results.append({
        'index': hit.get('id'),
        'distance': hit.get('score'),
        'metadata': hit.get('payload')
      })
    return results
  
  def query(self, query_text: str, top_k: int=5):
    query_emb = self.model.encode([query_text]).astype('float32')
    return self.search(query_emb, top_k=top_k)
    