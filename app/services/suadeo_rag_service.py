import json
import numpy as np
import requests
import os
import logging
from typing import List, Dict, Optional
from app.models.suadeo_models import RAGResponse, SuadeoRAGConfig
from app.services.suadeo_embeddings_service import OllamaEmbeddings
from app.services.suadeo_vector_store_service import FAISSVectorStore
from app.utils.catalog_utils import fetch_catalog_data
from app.utils.parse_catalog import parse_catalog_for_rag, save_rag_data, get_active_datasets_only, get_search_content_only, filter_by_catalog, filter_by_language
from app.exceptions.suadeo_exceptions import (
    SuadeoRAGException, IndexNotInitializedError, CatalogDataError
)

from app.utils.logging import get_logger

# Create a logger specific to this module
logger = get_logger("app.routers.suadeo_rag_services")

class SuadeoRAGService:
    """
    Main Suadeo RAG service that orchestrates all components
    """
    
    def __init__(self, config: SuadeoRAGConfig):
        """
        Initialize Suadeo RAG service
        
        Args:
            config: RAG configuration
        """
        self.config = config
        self.embeddings = OllamaEmbeddings(
            model=config.embedding_model,
            base_url=config.ollama_base_url
        )
        self.vector_store: Optional[FAISSVectorStore] = None
        self.rag_data: Dict = {'search_index': []}
        self.index_path = "suadeo_catalog_data/suadeo_index"
        self.rag_data_path = "suadeo_catalog_data/rag_data_all.json"
        self._is_initialized = False
        
        logger.info("Initialized Suadeo RAG Service")
    
    def _check_existing_index(self) -> bool:
        """
        Check if existing vector index and RAG data are available and valid
        
        Returns:
            True if valid existing index exists, False otherwise
        """
        try:
            # Check if all required files exist
            faiss_index_path = f"{self.index_path}.faiss"
            faiss_metadata_path = f"{self.index_path}.pkl"
            
            if not all([
                os.path.exists(faiss_index_path),
                os.path.exists(faiss_metadata_path),
                os.path.exists(self.rag_data_path)
            ]):
                logger.info("Some required index files are missing")
                return False
            
            # Check if files are not empty
            if (os.path.getsize(faiss_index_path) == 0 or 
                os.path.getsize(faiss_metadata_path) == 0 or
                os.path.getsize(self.rag_data_path) == 0):
                logger.warning("Some index files are empty")
                return False
            
            # Load and validate RAG data
            with open(self.rag_data_path, 'r', encoding='utf-8') as f:
                rag_data = json.load(f)
                
            if not rag_data.get('search_index') or len(rag_data['search_index']) == 0:
                logger.warning("RAG data file exists but contains no search index")
                return False
            
            logger.info(f"Found existing index with {len(rag_data['search_index'])} chunks")
            return True
            
        except Exception as e:
            logger.warning(f"Error checking existing index: {e}")
            return False
    
    def _load_rag_data(self) -> bool:
        """
        Load existing RAG data from file
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            with open(self.rag_data_path, 'r', encoding='utf-8') as f:
                self.rag_data = json.load(f)
                
            if not self.rag_data.get('search_index'):
                self.rag_data['search_index'] = []
                return False
                
            logger.info(f"Loaded {len(self.rag_data['search_index'])} chunks from existing RAG data")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load RAG data: {e}")
            self.rag_data = {'search_index': []}
            return False

    async def initialize(self, force_rebuild: bool = False):
        """
        Initialize the RAG system with catalog data from all endpoints
        
        Args:
            force_rebuild: Whether to force rebuild the index
        """
        try:
            # Check for existing index first
            has_existing_index = self._check_existing_index()
            
            if has_existing_index and not force_rebuild:
                logger.info("Found existing vector index, loading...")
                
                # Load existing RAG data
                if self._load_rag_data():
                    # Load vector store
                    await self._load_existing_index()
                    self._is_initialized = True
                    logger.info(f"RAG system loaded successfully with {len(self.rag_data['search_index'])} chunks")
                    return
                else:
                    logger.warning("Failed to load RAG data, will rebuild index")
            
            if force_rebuild:
                logger.info("Force rebuild requested, rebuilding index...")
            else:
                logger.info("No valid existing index found, building new index...")
            
            # Fetch and process catalog data
            endpoints = {
                "api/administration/userdata": "4",
                "api/administration/catalogs": "15",
                "api/BusinessGlossary": "11",
                "api/administration/connections": "5"
            }
            
            # Aggregate data from all endpoints
            all_search_chunks = []
            for endpoint_url, endpoint_type in endpoints.items():
                logger.info(f"Fetching catalog data from {endpoint_url}...")
                catalog_data = fetch_catalog_data(endpoint_url)
                
                if not catalog_data:
                    logger.warning(f"No data retrieved from {endpoint_url}")
                    continue
                
                # Process catalog data for RAG
                rag_data = self._process_catalog_data(catalog_data, endpoint_url, endpoint_type)
                
                if rag_data and 'search_index' in rag_data:
                    all_search_chunks.extend(rag_data['search_index'])
                    logger.info(f"Collected {len(rag_data['search_index'])} chunks from {endpoint_url}")
                
                # Save individual endpoint data
                safe_filename = endpoint_url.replace('/', '_')
                save_rag_data(rag_data, f"suadeo_catalog_data/rag_data_{safe_filename}.json")

            # Update self.rag_data with all collected chunks
            self.rag_data['search_index'] = all_search_chunks
            save_rag_data(self.rag_data, self.rag_data_path)
            
            # Build new vector index
            if all_search_chunks:
                logger.info("Building new vector index...")
                await self._build_new_index(all_search_chunks)
            else:
                raise CatalogDataError("No search chunks available to build index")
            
            self._is_initialized = True
            logger.info(f"RAG system initialized successfully with {len(all_search_chunks)} total chunks")
                
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise SuadeoRAGException(f"Initialization failed: {e}")
    
    def _process_catalog_data(self, catalog_data: Dict, source_endpoint: str,  source_type: str) -> Dict:
        """
        Process catalog data into RAG-friendly format
        
        Args:
            catalog_data: Raw catalog data from API
            source_endpoint: API endpoint from which the data was fetched
            source_type: Type identifier for the endpoint
            
        Returns:
            Processed RAG data
        """
        rag_data = parse_catalog_for_rag(catalog_data, source_endpoint, source_type)
        return rag_data
    
    async def _load_existing_index(self):
        """Load existing vector index"""
        try:
            sample_embedding = self.embeddings.embed_text("sample text for dimension")
            self.vector_store = FAISSVectorStore(dimension=len(sample_embedding))
            self.vector_store.load(self.index_path)
            logger.info("Existing vector index loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load existing index: {e}")
            raise SuadeoRAGException(f"Failed to load existing index: {e}")
    
    async def _build_new_index(self, search_chunks: List[Dict]):
        """Build new vector index from RAG data"""
        if not search_chunks:
            raise CatalogDataError("No search index data available")
        
        texts = [chunk['content'] for chunk in search_chunks]
        if not texts:
            raise CatalogDataError("No searchable content found in catalog data")
        
        logger.info(f"Creating embeddings for {len(texts)} chunks...")
        
        # Create embeddings for all texts
        embeddings = self.embeddings.embed_batch(texts, batch_size=1000)
        
        # Initialize vector store
        self.vector_store = FAISSVectorStore(dimension=embeddings.shape[1])
        
        # Add vectors with metadata
        self.vector_store.add_vectors(embeddings, search_chunks)
        
        # Save index
        self.vector_store.save(self.index_path)
        logger.info(f"New vector index built and saved with {len(search_chunks)} chunks")
    
    async def _update_index(self, new_chunks: List[Dict]):
        """Update existing vector index with new or updated chunks"""
        if not new_chunks:
            logger.info("No new chunks to update")
            return
        
        texts = [chunk['content'] for chunk in new_chunks]
        # texts = [chunk for chunk in new_chunks]
        if not texts:
            logger.warning("No searchable content in new chunks")
            return
        
        logger.info(f"Updating index with {len(texts)} new chunks...")
        
        # Create embeddings for new chunks
        embeddings = self.embeddings.embed_batch(texts, batch_size=1000)
        
        # Add new vectors to existing index
        self.vector_store.add_vectors(embeddings, new_chunks)
        
        # Save updated index
        self.vector_store.save(self.index_path)
    
    async def delete_entry(self, dataset_id: str):
        """
        Delete an entry from the vector store and RAG data by dataset_id
        
        Args:
            dataset_id: Unique identifier of the dataset to delete
        """
        if not self._is_initialized or not self.vector_store:
            raise IndexNotInitializedError("RAG system not initialized")
        
        try:
            # Remove from rag_data
            original_len = len(self.rag_data['search_index'])
            self.rag_data['search_index'] = [
                chunk for chunk in self.rag_data['search_index']
                if chunk.get('metadata', {}).get('dataset_id') != dataset_id
            ]
            removed_chunks = original_len - len(self.rag_data['search_index'])
            
            if removed_chunks == 0:
                logger.warning(f"No chunks found with dataset_id: {dataset_id}")
                return
            
            # Rebuild vector store from updated rag_data
            logger.info(f"Removed {removed_chunks} chunks with dataset_id: {dataset_id}. Rebuilding index...")
            await self._build_new_index(self.rag_data['search_index'])
            
            # Save updated rag_data
            save_rag_data(self.rag_data, self.rag_data_path)
            logger.info(f"Successfully deleted entries for dataset_id: {dataset_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete entry for dataset_id {dataset_id}: {e}")
            raise SuadeoRAGException(f"Deletion failed: {e}")
    
    def _assess_context_sufficiency(self, similarities: np.ndarray, retrieved_chunks: List[Dict]) -> bool:
        """
        Determine if retrieved context is sufficient to answer the query
        
        Args:
            similarities: Similarity scores from vector search
            retrieved_chunks: Retrieved content chunks
            
        Returns:
            Boolean indicating if context is sufficient
        """
        high_similarity_count = sum(1 for sim in similarities if sim >= self.config.min_similarity_threshold)
        content_types = set(chunk.get('content_type', 'unknown') for chunk in retrieved_chunks)
        enough_chunks = len(retrieved_chunks) >= self.config.min_context_chunks
        has_relevant_matches = high_similarity_count > 0
        has_sufficient_chunks = enough_chunks and high_similarity_count >= self.config.min_context_chunks
        
        logger.debug(f"Context assessment - High similarity: {high_similarity_count}, "
                    f"Content types: {len(content_types)}, Sufficient: {has_sufficient_chunks}")
        
        return has_relevant_matches and has_sufficient_chunks
    
    def _generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Generate response using Ollama LLM
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            
        Returns:
            Generated response
        """
        context_parts = []
        for chunk in context_chunks:
            metadata = chunk.get('metadata', {})
            context_part = f"[{chunk.get('content_type', 'info')}] {chunk['content']}"
            dataset_name = metadata.get("dataset_name", "Unknown")
            catalog = metadata.get("catalog", "Unknown")
            dataset_id = metadata.get("dataset_id", "Unknown")
            status = metadata.get("status", "Unknown")
            owner = metadata.get("owner", "Unknown")
            language = metadata.get("language", "Unknown")
            source_endpoint = metadata.get("source_endpoint", "Unknown")
            source_endpoint_type = metadata.get("source_endpoint_type", "Unknown")
            context_part += f"\n(Dataset: {dataset_name}, Catalog: {catalog}, ID: {dataset_id}, Status: {status}, Owner: {owner}, Language: {language}, Source Endpoint: {source_endpoint}, Source Endpoint type: {source_endpoint_type})"
            if domains := metadata.get("domains"):
                context_part += f"\nDomains: {', '.join(domains)}"
            context_parts.append(context_part)
        context = "\n\n".join(context_parts)
        
        logger.debug(f"Generated context: {context}\n\n")
        
        prompt = f"""
                Based on the following dataset documentation and metadata, answer the user's question in detail way. 

                Context:
                {context}

                Question: {query}

                Instructions:
                - Use only the information provided in the context
                - If insufficient, explicitly say so
                - Mention dataset names, languages, source endpoints, dataset id and details (e.g., catalog, owner, status, source endpoint) when possible
                - Be concise but detailed.
                - If you find even 1 chunk with above 70% similarity use that and fully answer.

                Answer:"""
        
        try:
            response = requests.post(
                f"{self.config.ollama_base_url}/api/generate",
                json={
                    "model": self.config.llm_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "Sorry, I couldn't generate a response.")
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, there was an error generating the response."
        
        
    async def search(self, query: str, k: int = 5) -> RAGResponse:
        """
        Main search method
        
        Args:
            query: User question
            k: Number of chunks to retrieve
            
        Returns:
            RAGResponse with answer and metadata
        """
        if not self._is_initialized:
            raise IndexNotInitializedError("RAG system not initialized. Call initialize first.")
        
        if not self.vector_store or self.vector_store.is_empty:
            raise IndexNotInitializedError("Vector store is empty or not loaded")
        
        logger.info(f"Processing search query: {query}")
        
        try:
            query_embedding = self.embeddings.embed_text(query)
            similarities, indices, retrieved_chunks = self.vector_store.search(query_embedding, k=k)
            
            # Deduplicate chunks based on chunk ID
            unique_chunks = []
            seen_ids = set()
            for sim, chunk in zip(similarities, retrieved_chunks):
                chunk_id = chunk.get('id', '')
                if chunk_id not in seen_ids:
                    unique_chunks.append((sim, chunk))
                    seen_ids.add(chunk_id)
            
            # Separate similarities and chunks after deduplication
            if unique_chunks:
                similarities, retrieved_chunks = zip(*unique_chunks)
                similarities = np.array(similarities)
                retrieved_chunks = list(retrieved_chunks)
            else:
                similarities = np.array([])
                retrieved_chunks = []
            
            has_sufficient_context = self._assess_context_sufficiency(similarities, retrieved_chunks)
            
            relevant_chunks = []
            relevant_sources = []
            for sim, chunk in zip(similarities, retrieved_chunks):
                if sim >= self.config.min_similarity_threshold:
                    relevant_chunks.append(chunk)
                    catalog_value = chunk.get('metadata', {}).get('catalog') or 'Unknown'  # Ensure None is replaced
                    source_endpoint = chunk.get('metadata', {}).get('source_endpoint', 'Unknown')
                    source_endpoint_type = chunk.get('metadata', {}).get('source_endpoint_type', 'Unknown')
                    relevant_sources.append({
                        'dataset_name': chunk.get('metadata', {}).get('dataset_name', 'Unknown'),
                        'content_type': chunk.get('content_type', 'unknown'),
                        'similarity': float(sim),
                        'catalog': catalog_value,
                        'source_endpoint': source_endpoint,
                        "source_endpoint_type":source_endpoint_type,
                        'chunk': chunk
                    })
            
            if has_sufficient_context and relevant_chunks:
                answer = self._generate_response(query, relevant_chunks)
                confidence = float(np.mean([sim for sim in similarities if sim >= self.config.min_similarity_threshold]))
            else:
                
                answer = "I don't have sufficient information in our catalog to answer this question accurately."
                confidence = 0.0
            
            logger.debug(f"Returning {len(relevant_sources)} unique sources for query: {query}")
            return RAGResponse(
                answer=answer,
                confidence=confidence,
                sources=relevant_sources,
                has_sufficient_context=has_sufficient_context,
                retrieved_chunks=retrieved_chunks
            )
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SuadeoRAGException(f"Search operation failed: {e}")
        
    async def refresh_data(self):
        """Refresh catalog data and rebuild index"""
        logger.info("Refreshing catalog data...")
        await self.initialize(force_rebuild=True)
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            "initialized": self._is_initialized,
            "vector_store_size": self.vector_store.index.ntotal if self.vector_store else 0,
            "rag_data_chunks": len(self.rag_data.get('search_index', [])),
            "embedding_model": self.config.embedding_model,
            "llm_model": self.config.llm_model,
            "index_path": self.index_path,
            "has_existing_index": self._check_existing_index()
        }