"""Model definitions and data structures for the RAGS system."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Define our dependencies
@dataclass
class RAGDependencies:
    chroma_client: chromadb.Client
    collection_name: str
    embedding_model: SentenceTransformer

# Create a model with a custom provider pointing to a local server
def create_llm_model(base_url="http://127.0.0.1:8080/v1", api_key="not-needed", model_name="gpt-4o"):
    local_provider = OpenAIProvider(
        base_url=base_url,
        api_key=api_key
    )
    
    return OpenAIModel(
        model_name,
        provider=local_provider
    )
