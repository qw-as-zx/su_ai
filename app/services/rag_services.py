"""
RAG Service Manager - Centralized RAG service instance management
"""
from typing import Optional
from app.services.suadeo_rag_service import SuadeoRAGService
from app.models.suadeo_models import SuadeoRAGConfig
from app.utils.logging import get_logger

logger = get_logger("app.services.rag_manager")

class RAGServiceManager:
    """Singleton manager for RAG service instance"""
    
    def __init__(self):
        self._rag_service: Optional[SuadeoRAGService] = None
        self._is_initializing = False
    
    async def initialize_service(self, force_rebuild: bool = False) -> bool:
        """
        Initialize RAG service with vector DB check
        
        Args:
            force_rebuild: Whether to force rebuild the index
            
        Returns:
            True if successful, False otherwise
        """
        if self._is_initializing:
            logger.warning("RAG service initialization already in progress")
            return False
            
        if self._rag_service is not None and self._rag_service._is_initialized and not force_rebuild:
            logger.info("RAG service already initialized")
            return True
        
        try:
            self._is_initializing = True
            logger.info("Initializing Suadeo RAG service...")
            
            config = SuadeoRAGConfig()
            rag_service = SuadeoRAGService(config)
            
            # Initialize with automatic vector DB detection
            await rag_service.initialize(force_rebuild=force_rebuild)
            
            self._rag_service = rag_service
            logger.info("Suadeo RAG service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            self._rag_service = None
            return False
        finally:
            self._is_initializing = False
    
    def get_service(self) -> Optional[SuadeoRAGService]:
        """Get the RAG service instance"""
        return self._rag_service
    
    def is_ready(self) -> bool:
        """Check if RAG service is ready"""
        return (self._rag_service is not None and 
                self._rag_service._is_initialized and 
                not self._is_initializing)
    
    def is_initializing(self) -> bool:
        """Check if RAG service is currently initializing"""
        return self._is_initializing
    
    async def refresh_service(self):
        """Refresh the RAG service by force rebuilding"""
        if self._rag_service:
            await self._rag_service.refresh_data()
        else:
            await self.initialize_service(force_rebuild=True)

# Global instance
rag_manager = RAGServiceManager()

