from pydantic import BaseModel, Field,  validator
from typing import List, Dict, Any, Optional

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: float = 0

class ChatRequest(BaseModel):
    session_id: str
    query: str

class EchoRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    sources: List[Dict[str, Any]] = []
    new_files_processed: bool = False

class FileUploadResponse(BaseModel):
    message: str

class AudioTranscript(BaseModel):
    text: str

class ErrorResponse(BaseModel):
    detail: str


## web search agent
class WebSearchRequest(BaseModel):
    query: str
    search_context_size: str = "medium"

    @validator("search_context_size")
    def validate_context_size(cls, v):
        valid_context_sizes = {"high", "medium", "low"}
        if v not in valid_context_sizes:
            raise ValueError(f"search_context_size must be one of {valid_context_sizes}")
        return v

class WebSearchResponse(BaseModel):
    answer: str
    citations: list[dict]

class MapSearchResponse(BaseModel):
    geojson: dict
    citations: list[dict]


## SQL Agent
class SQLQueryRequest(BaseModel):
    """Request schema for SQL Agent queries"""
    query: str = Field(..., description="Natural language query to be converted to SQL")
    database_uri: Optional[str] = Field(None, description="Database URI to connect to (optional)")

class SQLResponse(BaseModel):
    """Response schema for SQL Agent results"""
    answer: str = Field(..., description="Natural language answer to the query")
    generated_sql: Optional[str] = Field(None, description="The SQL query that was generated")
    execution_results: Optional[Any] = Field(None, description="Raw results from executing the SQL query")
    steps: Optional[List[Dict[str, Any]]] = Field(None, description="Steps taken by the agent")