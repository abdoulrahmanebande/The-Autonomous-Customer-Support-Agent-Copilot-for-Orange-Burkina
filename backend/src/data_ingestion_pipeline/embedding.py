from src.data_ingestion_pipeline.data_loader import load_all_documents
from sentence_transformers import SentenceTransformer  # From hugging face
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Any
from src.logger import logging
from src.exception import CustomeException

class EmbeddingPipeline:
  def __init__(
    self, 
    chunk_size: int=1000,
    chunk_overlap: int=200,
    model_name :str='intfloat/multilingual-e5-base'
  ):
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap
    self.model = SentenceTransformer(model_name)
    logging.info(f"Loading embedding model: {model_name}")
    
  def chunk_documents(self, documents: List[Any]) -> List[Any]:
    splitter = RecursiveCharacterTextSplitter(
      chunk_size = self.chunk_size,
      chunk_overlap = self.chunk_overlap,
      separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    logging.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    
    return chunks
  
  def embed_chunks(self, chunks: List[Any]) -> List[Any]:
    texts = [chunk.page_content for chunk in chunks]
    logging.info(f'Generating embeddings for {len(texts)} chunks.')
    
    embeddings = self.model.encode(texts, show_progress_bar=True).astype('float32')
    logging.info(f"Embedding shape: {embeddings.shape}")
    
    return embeddings