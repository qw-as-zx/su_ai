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
        self.rag_data: Optional[Dict] = None
        self.index_path = "suadeo_catalog_data/suadeo_index"
        self._is_initialized = False
        
        logger.info("Initialized Suadeo RAG Service")
    
    async def initialize(self, force_rebuild: bool = False):
        """
        Initialize the RAG system with catalog data
        
        Args:
            force_rebuild: Whether to force rebuild the index
        """
        try:
            # Fetch fresh catalog data
            logger.info("Fetching catalog data...")
            catalog_data = fetch_catalog_data()
            
            if not catalog_data:
                raise CatalogDataError("Failed to fetch catalog data")
            
            # Process catalog data for RAG (you'll need to implement this based on your parse_catalogue.py)
            self.rag_data = self._process_catalog_data(catalog_data)
            save_rag_data(self.rag_data)

            ## Some testing of the rag data
            # Example of extracting content for embeddings
            logger.info("\n\t\t=== Content for Vector Embeddings ===")
            contents = get_search_content_only(self.rag_data)
            for i, content in enumerate(contents):
                logger.info(f"{i+1}. {content[:100]}...")
                break
            

            ## Extra checks

            logger.info("\n\t\t=== Filter German Descriptions only ===")
            contents = filter_by_language(self.rag_data,'de')
            for i, content in enumerate(contents):
                logger.info(f"{i+1}. {content['content'][:100]}...")
                break

            logger.info("\n\t\t=== Active Datasets only ===")
            contents = get_active_datasets_only(self.rag_data)
            for i, content in enumerate(contents):
                logger.info(f"{i+1}. {content['name'][:100]}...")  # or use 'id' or 'key'
                break


            logger.info("\n\t\t=== Filter Content Only ===")
            contents = filter_by_catalog(self.rag_data,"JOP Paris 2024")
            for i, content in enumerate(contents):
                logger.info(f"{i+1}. {content['name'][:100]}...")
                break
                
                     
            # Load or build vector index
            if os.path.exists(f"{self.index_path}.faiss") and not force_rebuild:
                logger.info("Loading existing vector index...")
                await self._load_existing_index()
            else:
                logger.info("Building new vector index...")
                await self._build_new_index()
            
            self._is_initialized = True
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise SuadeoRAGException(f"Initialization failed: {e}")
    
    def _process_catalog_data(self, catalog_data: Dict) -> Dict:
        """
        Process catalog data into RAG-friendly format
        Note: This needs to be implemented based on your parse_catalogue.py logic
        
        Args:
            catalog_data: Raw catalog data from API
            
        Returns:
            Processed RAG data
        """
        rag_data = parse_catalog_for_rag(catalog_data)
        return rag_data
    
    async def _load_existing_index(self):
        """Load existing vector index"""
        # Get embedding dimension from first available text or use default
        sample_embedding = self.embeddings.embed_text("sample text for dimension")
        
        self.vector_store = FAISSVectorStore(dimension=len(sample_embedding))
        self.vector_store.load(self.index_path)
    
    async def _build_new_index(self):
        """Build new vector index from RAG data"""
        if not self.rag_data or not self.rag_data.get('search_index'):
            raise CatalogDataError("No search index data available")
        
        # Extract all searchable content
        search_chunks = self.rag_data['search_index']
        texts = [chunk['content'] for chunk in search_chunks]
        
        if not texts:
            raise CatalogDataError("No searchable content found in catalog data")
        
        logger.info(f"Creating embeddings for {len(texts)} chunks...")
        
        # Create embeddings for all texts
        embeddings = self.embeddings.embed_batch(texts, batch_size=50)
        
        # Initialize vector store
        self.vector_store = FAISSVectorStore(dimension=embeddings.shape[1])
        
        # Add vectors with metadata
        self.vector_store.add_vectors(embeddings, search_chunks)
        
        # Save index
        self.vector_store.save(self.index_path)
    
    def _assess_context_sufficiency(self, similarities: np.ndarray, retrieved_chunks: List[Dict]) -> bool:
        """
        Determine if retrieved context is sufficient to answer the query
        
        Args:
            similarities: Similarity scores from vector search
            retrieved_chunks: Retrieved content chunks
            
        Returns:
            Boolean indicating if context is sufficient
        """
        # Check for high-similarity matches
        high_similarity_count = sum(1 for sim in similarities if sim >= self.config.min_similarity_threshold)
        
        # Check for diverse content types
        content_types = set(chunk.get('content_type', 'unknown') for chunk in retrieved_chunks)
        
        # Check for enough chunks
        enough_chunks = len(retrieved_chunks) >= self.config.min_context_chunks
        
        # Determine sufficiency
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
        # Build context from chunks
        context_parts = []
        # for chunk in context_chunks:
        #     context_part = f"[{chunk.get('content_type', 'info')}] {chunk['content']}"
            
        #     if chunk.get('metadata', {}).get('dataset_name'):
        #         context_part += f" (Dataset: {chunk['metadata']['dataset_name']})"
        #         context_part+= f"All Metadata: {chunk.get('metadata','No Metadata avialable')}"
        #     context_parts.append(context_part)
        ## Updated Version
        for chunk in context_chunks:
            metadata = chunk.get('metadata', {})
            
            context_part = f"[{chunk.get('content_type', 'info')}] {chunk['content']}"

            # Include readable metadata
            dataset_name = metadata.get("dataset_name", "Unknown")
            catalog = metadata.get("catalog", "Unknown")
            dataset_id = metadata.get("dataset_id", "Unknown")
            status = metadata.get("status", "Unknown")
            owner = metadata.get("owner", "Unknown")
            language = metadata.get("language", "Unknown")

            context_part += f"\n(Dataset: {dataset_name}, Catalog: {catalog}, ID: {dataset_id}, Status: {status}, Owner: {owner}, Language: {language})"

            # Optionally include domains or other nested metadata
            if domains := metadata.get("domains"):
                context_part += f"\nDomains: {', '.join(domains)}"

            context_parts.append(context_part)
        context = "\n\n".join(context_parts)

        
        logger.debug(f"Generated context: {context}\n\n")
        
        # Create prompt for LLM
        # prompt = f"""Based on the following context from our data catalog, please answer the user's question.

        #             Context:
        #             {context}

        #             Question: {query}

        #             Instructions:
        #             - Answer based only on the provided context
        #             - If the context doesn't contain enough information, say so clearly
        #             - Be specific and mention relevant datasets when applicable
        #             - Keep the answer concise but informative

        #             Answer:"""

        ## Updated prompt
        prompt = f"""Based on the following dataset documentation and metadata, answer the user's question.

                Context:
                {context}

                Question: {query}

                Instructions:
                - Use only the information provided in the context
                - If insufficient, explicitly say so
                - Mention dataset names and details (e.g., catalog, owner, status) when possible
                - Be concise but detailed

                Answer:"""



        # Call Ollama LLM
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
            raise IndexNotInitializedError("RAG system not initialized. Call initialize() first.")
        
        if not self.vector_store or self.vector_store.is_empty:
            raise IndexNotInitializedError("Vector store is empty or not loaded")
        
        logger.info(f"Processing search query: {query}")
        
        try:
            # Step 1: Create query embedding
            query_embedding = self.embeddings.embed_text(query)
            
            # Step 2: Search vector store
            similarities, indices, retrieved_chunks = self.vector_store.search(query_embedding, k=k)
            
            # Step 3: Assess context sufficiency
            has_sufficient_context = self._assess_context_sufficiency(similarities, retrieved_chunks)
            
            # Step 4: Filter relevant chunks
            relevant_chunks = []
            relevant_sources = []
            
            for sim, chunk in zip(similarities, retrieved_chunks):
                if sim >= self.config.min_similarity_threshold:
                    relevant_chunks.append(chunk)
                    catalog_value = chunk.get('metadata', {}).get('catalog') or 'Unknown'
                    relevant_sources.append({
                        'dataset_name': chunk.get('metadata', {}).get('dataset_name', 'Unknown'),
                        'content_type': chunk.get('content_type', 'unknown'),
                        'similarity': float(sim),
                        'catalog': catalog_value,  #chunk.get('metadata', {}).get('catalog', 'Unknown')
                        # 'chunk_text': chunk.get('content', '')  # ADD THE CHUNK CONTENT HERE
                        'chunk': chunk
                    })
            
            # Step 5: Generate response if we have sufficient context
            if has_sufficient_context and relevant_chunks:
                answer = self._generate_response(query, relevant_chunks)
                confidence = float(np.mean([sim for sim in similarities if sim >= self.config.min_similarity_threshold]))
            else:
                answer = "I don't have sufficient information in our catalog to answer this question accurately."
                confidence = 0.0
            
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
            "embedding_model": self.config.embedding_model,
            "llm_model": self.config.llm_model,
            "index_path": self.index_path
        }