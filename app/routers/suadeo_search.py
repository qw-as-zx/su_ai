from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from app.utils.query_classification import generate_multilingual_rag_response, create_optimized_english_search_query, detect_language, classify_query, create_greeting_response, classify_relevance, create_irrelevant_response

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

from app.services.rag_services  import rag_manager
async def get_rag_service() -> SuadeoRAGService:
    """Dependency to get RAG service instance"""
    
    # Check if service is ready
    if rag_manager.is_ready():
        return rag_manager.get_service()
    
    # Check if service is currently initializing
    if rag_manager.is_initializing():
        raise HTTPException(
            status_code=503,
            detail="RAG service is currently initializing. Please try again in a few moments."
        )
    
    # Try to initialize if not ready and not initializing
    try:
        logger.warning("RAG service not initialized during startup, attempting initialization...")
        success = await rag_manager.initialize_service()
        
        if success and rag_manager.is_ready():
            logger.info("RAG service initialized successfully on demand")
            return rag_manager.get_service()
        else:
            raise HTTPException(
                status_code=500,
                detail="RAG service initialization failed"
            )
            
    except Exception as e:
        logger.error(f"Failed to initialize RAG service on demand: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"RAG service initialization failed: {str(e)}"
        )

@router.post("/suadeo_search", response_model=SuadeoSearchResponse)
async def suadeo_search(
    request: SuadeoSearchRequest,
    rag_service: SuadeoRAGService = Depends(get_rag_service)
):
    """
    Search the Suadeo catalog using RAG with guardrails
    
    This endpoint performs semantic search across your Suadeo data catalog
    and provides AI-generated responses with source references.
    """
    try:

        logger.info(f"Received search request: {request.query}")
        
        # Detect the language of the query
        detected_language = await detect_language(request.query)
        logger.info(f"Detected language: {detected_language}")
        
        # First, classify if the query is a greeting
        greeting_classification = await classify_query(request.query)
        logger.info(f"Greeting classification: is_greeting={greeting_classification.is_greeting}, confidence={greeting_classification.confidence}")
        
        # If it's a greeting with high confidence, return welcome message
        if greeting_classification.is_greeting and greeting_classification.confidence > 0.7:
            logger.info("Responding with greeting message")
            return await create_greeting_response(request.query)
        
        # If not a greeting, check if it's relevant to Suadeo catalog
        relevance_classification = await classify_relevance(request.query)
        logger.info(f"Relevance classification: is_relevant={relevance_classification.is_relevant}, confidence={relevance_classification.confidence}")
        
        # If not relevant with high confidence, return guidance message
        if not relevance_classification.is_relevant and relevance_classification.confidence >= 0.7:
            logger.info("Query deemed irrelevant to Suadeo catalog")
            return await create_irrelevant_response(request.query, relevance_classification.suggested_topics)
        
        # If borderline relevance (low confidence), log but proceed with search
        if relevance_classification.confidence < 0.6:
            logger.warning(f"Low confidence relevance classification: {relevance_classification.confidence}")
        
        # Create optimized English search query for FAISS vector search
        optimized_search_query = await create_optimized_english_search_query(request.query, detected_language)
        logger.info(f"Using optimized search query: '{optimized_search_query}'")
        
        # Proceed with RAG search using optimized English query
        logger.info("Proceeding with RAG search")
        
        # Create a modified request with the optimized query for RAG search
        optimized_request = SuadeoSearchRequest(
            query=optimized_search_query,
            k=request.k
        )
        
        # Perform RAG search with optimized English query
        rag_response = await rag_service.search(
            query=optimized_request.query,
            k=optimized_request.k
        )
        
        logger.info(f"RAG search completed successfully\nRAG Results: {rag_response}\n\nNow Generating the Multilingual RAG Response: \n")
        # # Generate response in the original language using RAG results
        logger.info("Generating multilingual RAG response...")
        multilingual_answer = await generate_multilingual_rag_response(
            original_query=request.query,
            rag_results={
                'answer': rag_response.answer,
                # 'sources': rag_response.sources, 
                'sources': [
                    {
                        'dataset_name': source.get('dataset_name'),
                        'content_type': source.get('content_type'),
                        'similarity': source.get('similarity'),
                        'catalog': source.get('catalog'),
                        'source_endpoint': source.get('source_endpoint'),
                        'source_endpoint_type': source.get('source_endpoint_type'),
                        'chunk': source.get('chunk')
                    }
                    for source in rag_response.sources
                ],

                'confidence': rag_response.confidence,
                'has_sufficient_context': rag_response.has_sufficient_context
            },
            detected_language=detected_language,
            optimized_query=optimized_search_query
        )

        logger.info(f"Generated multilingual RAG response in {detected_language}\n{multilingual_answer}\n\nConverting this to api response format...")
        # Convert to API response format
        sources = [
            SearchSource(
                dataset_name=source['dataset_name'],
                content_type=source['content_type'],
                similarity=source['similarity'],
                catalog=source['catalog'] or 'Unknown',
                source_endpoint=source['source_endpoint'],
                source_endpoint_type=source['source_endpoint_type'],
                chunk=source['chunk']
            )
            for source in rag_response.sources
        ]
        ## TODO: Change the response structure according to the required need...
        
        response = SuadeoSearchResponse(
            answer=multilingual_answer, #rag_response.answer,#,  # Use the multilingual response
            confidence=rag_response.confidence,
            sources=sources,
            has_sufficient_context=rag_response.has_sufficient_context,
            query=request.query,  # Keep original query for reference
            total_chunks_found=len(rag_response.retrieved_chunks)
        )
        
        logger.info(f"Search completed successfully. Confidence: {response.confidence:.2f}, Language: {detected_language}\n\nHere is the final Response")
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