from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.utils.config import settings

class SuadeoSearchRequest(BaseModel):
    """Request model for Suadeo search"""
    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    k: int = Field(default=5, description="Number of results to return", ge=1, le=20)
    min_similarity: float = Field(default=0.7, description="Minimum similarity threshold", ge=0.0, le=1.0)

class SearchSource(BaseModel):
    """Individual search result source"""
    dataset_name: str
    content_type: str
    similarity: float
    catalog: str
    source_endpoint: str
    chunk: Dict

class SuadeoSearchResponse(BaseModel):
    """Response model for Suadeo search"""
    answer: str
    confidence: float
    sources: List[SearchSource]
    has_sufficient_context: bool
    query: str
    total_chunks_found: int

class SuadeoRAGConfig(BaseModel):
    """Configuration for Suadeo RAG system"""
    embedding_model: str = settings.embedding_model
    ollama_base_url: str = settings.ollama_base_url
    llm_model: str = settings.ollama_model
    min_similarity_threshold: float = settings.min_similarity_threshold
    min_context_chunks: int = settings.min_context_chunks
    index_rebuild_threshold: int = settings.index_rebuild_threshold

@dataclass
class RAGResponse:
    """Internal RAG response structure"""
    answer: str
    confidence: float
    sources: List[Dict]
    has_sufficient_context: bool
    retrieved_chunks: List[Dict]