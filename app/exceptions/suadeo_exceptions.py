class SuadeoRAGException(Exception):
    """Base exception for Suadeo RAG operations"""
    pass

class OllamaConnectionError(SuadeoRAGException):
    """Raised when cannot connect to Ollama server"""
    pass

class EmbeddingGenerationError(SuadeoRAGException):
    """Raised when embedding generation fails"""
    pass

class VectorStoreError(SuadeoRAGException):
    """Raised when vector store operations fail"""
    pass

class CatalogDataError(SuadeoRAGException):
    """Raised when catalog data is invalid or missing"""
    pass

class IndexNotInitializedError(SuadeoRAGException):
    """Raised when trying to search without initialized index"""
    pass