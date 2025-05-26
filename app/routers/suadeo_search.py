from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
import logging
from app.models.suadeo_models import (
    SuadeoSearchRequest, 
    SuadeoSearchResponse, 
    SearchSource,
    SuadeoRAGConfig
)

from datetime import datetime

from app.services.suadeo_rag_service import SuadeoRAGService
from app.exceptions.suadeo_exceptions import (
    SuadeoRAGException,
    IndexNotInitializedError,
    OllamaConnectionError
)

from app.utils.logging import get_logger

# Create a logger specific to this module
logger = get_logger("app.routers.suadeo_search")

# Initialize router
router = APIRouter(prefix="", tags=["Suadeo search"])

# Global RAG service instance (you might want to use dependency injection)
_rag_service: SuadeoRAGService = None

async def get_rag_service() -> SuadeoRAGService:
    """Dependency to get RAG service instance"""
    global _rag_service
    
    if _rag_service is None:
        # Initialize with default config
        config = SuadeoRAGConfig()
        _rag_service = SuadeoRAGService(config)
        
        try:
            await _rag_service.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"RAG service initialization failed: {str(e)}"
            )
    
    return _rag_service

@router.post("/suadeo_search", response_model=SuadeoSearchResponse)
async def suadeo_search(
    request: SuadeoSearchRequest,
    rag_service: SuadeoRAGService = Depends(get_rag_service)
):
    """
    Search the Suadeo catalog using RAG
    
    This endpoint performs semantic search across your Suadeo data catalog
    and provides AI-generated responses with source references.
    """
    try:
        logger.info(f"Received search request: {request.query}")
        
        # Perform RAG search
        rag_response = await rag_service.search(
            query=request.query,
            k=request.k
        )
        
        # Convert to API response format
        sources = [
            SearchSource(
                dataset_name=source['dataset_name'],
                content_type=source['content_type'],
                similarity=source['similarity'],
                catalog=source['catalog'],
                source_endpoint=source['source_endpoint'],
                chunk=source['chunk']  # or chunk_text=source['chunk']['content']
                # chunk_text=source['chunk_text']  # MAP THIS TOO
            )
            for source in rag_response.sources
        ]
        
        response = SuadeoSearchResponse(
            answer=rag_response.answer,
            confidence=rag_response.confidence,
            sources=sources,
            has_sufficient_context=rag_response.has_sufficient_context,
            query=request.query,
            total_chunks_found=len(rag_response.retrieved_chunks)
        )
        
        logger.info(f"Search completed successfully. Confidence: {response.confidence:.2f}")
        return response
        
    except IndexNotInitializedError as e:
        logger.error(f"Index not initialized: {e}")
        raise HTTPException(
            status_code=503,
            detail="Search index is not ready. Please try again in a few moments."
        )
        
    except OllamaConnectionError as e:
        logger.error(f"Ollama connection error: {e}")
        raise HTTPException(
            status_code=503,
            detail="AI service is currently unavailable. Please try again later."
        )
        
    except SuadeoRAGException as e:
        logger.error(f"RAG operation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search operation failed: {str(e)}"
        )
        
    except Exception as e:
        logger.error(f"Unexpected error during search: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during search"
        )

@router.get("/suadeo_search_status")
async def get_system_status(
    rag_service: SuadeoRAGService = Depends(get_rag_service)
):
    """
    Get current system status and health information
    """
    try:
        status = rag_service.get_system_status()
        return {
            "status": "healthy" if status["initialized"] else "initializing",
            "details": status,
            "message": "System is ready" if status["initialized"] else "System is initializing"
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            "status": "unhealthy",
            "details": {"error": str(e)},
            "message": "System is experiencing issues"
        }

@router.post("/suadeo_search_refresh")
async def refresh_catalog_data(
    background_tasks: BackgroundTasks,
    rag_service: SuadeoRAGService = Depends(get_rag_service)
):
    """
    Refresh catalog data and rebuild search index
    
    This operation runs in the background to avoid blocking the API
    """
    try:
        # Add refresh task to background
        background_tasks.add_task(rag_service.refresh_data)
        
        return {
            "message": "Catalog refresh initiated",
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error initiating catalog refresh: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to initiate catalog refresh"
        )

@router.get("/sudeo_search_health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "ok",
        "service": "Suadeo Search API",
        "timestamp": datetime.utcnow().isoformat() + "Z" #"2024-01-01T00:00:00Z"  # You might want to use actual timestamp
    }