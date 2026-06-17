from langchain_community.document_loaders import PyMuPDFLoader, JSONLoader
from typing import List, Any
from pathlib import Path
from src.logger import logging
from src.exception import CustomeException
import sys 

def metadata_func(record: dict, metadata: dict) -> dict:
  """
    Custom metadata extractor to ensure RAG filters can see
    the exact sources and titles of your synthetic entries.
  """
  metadata['title'] = record.get('title', 'Untitled Support Doc')
  metadata['url'] = record.get('url', 'Unknown Source')
  
  return metadata

def load_all_documents(data_dir: str) -> List[Any]:
  data_path = Path(data_dir).resolve()
  logging.info(f"Data path: {data_path}")
  
  documents = []
  
  # PDF files
  pdf_files = list(data_path.glob("**/*.pdf"))
  logging.info(f"Found {len(pdf_files)} pdf files: {[str(pdf_file) for pdf_file in pdf_files]}")
  
  for pdf_file in pdf_files:
    try:
      pdf_loader = PyMuPDFLoader(pdf_file)
      pdf_loaded = pdf_loader.load()
      logging.info(f"Found {len(pdf_loaded)} pdf docs from {pdf_file}")
      
      documents.extend(pdf_loaded)
    except Exception as e:
      logging.info(f"Error occured: {e}")
      raise CustomeException(e, sys)
    
  # JSON File 
  json_files = list(data_path.glob('**/*.json'))
  logging.info(f"Found {len(json_files)} json files: {[str(json_file) for json_file in json_files]}")
  
  for json_file in json_files:
    try:
      # jq_schema=".[]" splits the root JSON array into individual elements.
      # content_key="content" targets the specific text field for embeddings.
      json_loader = JSONLoader(
        file_path=json_file,
        jq_schema='.[]',
        content_key="content",
        metadata_func=metadata_func
      )
      json_loaded = json_loader.load()
      logging.info(f"Found {len(json_loaded)} json docs from {json_file}")
      
      documents.extend(json_loaded)
    except Exception as e:
      logging.info(f"Error occured: {e}")
      raise CustomeException(e, sys)
    
  return documents
    

  """
jq_schema=".[]": Tells the system to loop through the array elements rather 
than treating the whole dataset as one big block. If you use '.' it eill treat
the whole JSON as one doc which is not good. If you json data was 
{
  "data": [
    {
      "url": "https://...",
      "title": "...",
      "content": "..."
    }
  ]
} 
In that case, using jq_schema=".data[]" is exactly right.

content_key="content": Ensures that only your core French explanation strings are vectorized, 
ignoring structural characters like commas or brackets.

metadata_func: Populates the underlying metadata dictionary so that when your agent retrieves 
an answer, it can print the exact URL reference matching the source file.
  """
    