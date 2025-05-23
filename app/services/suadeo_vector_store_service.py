import numpy as np
import faiss
import pickle
import os
import logging
from typing import List, Dict, Tuple
from app.exceptions.suadeo_exceptions import VectorStoreError

from app.utils.logging import get_logger

# Create a logger specific to this module
logger = get_logger("app.routers.suadeo_vector_store_services")

class FAISSVectorStore:
    """
    FAISS vector store for efficient similarity search
    """
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initialize FAISS vector store
        
        Args:
            dimension: Embedding dimension
            index_type: Type of FAISS index ('flat' for exact search, 'ivf' for approximate)
        """
        self.dimension = dimension
        self.index_type = index_type
        
        try:
            # Create FAISS index
            if index_type == "flat":
                self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
            elif index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            
            self.metadata = []
            self.is_trained = False
            
            logger.info(f"Initialized FAISS {index_type} index with dimension {dimension}")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize vector store: {e}")
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """
        Add vectors to the index
        
        Args:
            vectors: numpy array of embeddings
            metadata: List of metadata dicts corresponding to each vector
        """
        try:
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(vectors)
            
            # Train index if needed (for IVF indexes)
            if self.index_type == "ivf" and not self.is_trained:
                self.index.train(vectors)
                self.is_trained = True
            
            # Add vectors to index
            self.index.add(vectors)
            self.metadata.extend(metadata)
            
            logger.info(f"Added {len(vectors)} vectors to index. Total: {self.index.ntotal}")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to add vectors: {e}")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query embedding
            k: Number of results to return
            
        Returns:
            Tuple of (similarities, indices, metadata)
        """
        try:
            # Normalize query vector
            query_vector = query_vector.reshape(1, -1)
            faiss.normalize_L2(query_vector)
            
            # Search
            similarities, indices = self.index.search(query_vector, k)
            
            # Get metadata for results
            result_metadata = [self.metadata[idx] for idx in indices[0] if idx < len(self.metadata)]
            
            return similarities[0], indices[0], result_metadata
            
        except Exception as e:
            raise VectorStoreError(f"Search failed: {e}")
    
    def save(self, filepath: str):
        """Save index and metadata to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save metadata
            with open(f"{filepath}_metadata.pkl", 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'dimension': self.dimension,
                    'index_type': self.index_type,
                    'is_trained': self.is_trained
                }, f)
            
            logger.info(f"Saved vector store to {filepath}")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to save vector store: {e}")
    
    def load(self, filepath: str):
        """Load index and metadata from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load metadata
            with open(f"{filepath}_metadata.pkl", 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.dimension = data['dimension']
                self.index_type = data['index_type']
                self.is_trained = data['is_trained']
            
            logger.info(f"Loaded vector store from {filepath}")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to load vector store: {e}")
    
    @property
    def is_empty(self) -> bool:
        """Check if vector store is empty"""
        return self.index.ntotal == 0