import os
import uuid
import time
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_ollama import OllamaEmbeddings
from app.utils.config import settings
from app.exceptions.custom_exceptions import VectorStoreError
from app.services.llm_service import llm_service 

from app.utils.logging import get_logger

# Create a logger specific to this module
logger = get_logger("app.services.vector_store")


class VectorStoreService:
    def __init__(self):
        logger.info(f"Initializing VectorStoreService with embedding model: {settings.embedding_model}")
        
        # Initialize embeddings based on configuration
        if settings.embedding_provider == "ollama":
            self.embeddings = llm_service.get_embeddings()  

            logger.info(f"Using Ollama embeddings with model: {settings.embedding_model}")
        elif settings.embedding_provider == "huggingface":
            self.embeddings = llm_service.get_embeddings()  

            logger.info(f"Using HuggingFace embeddings with model: {settings.embedding_model}")
        else:
            raise VectorStoreError(f"Unsupported embedding provider: {settings.embedding_provider}")
        
        # Regular text splitter for non-tabular data
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=300,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Specialized text splitter for tabular data with larger chunk size
        self.tabular_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,  # Larger chunk size for tabular data
            chunk_overlap=500,
            separators=["\n\n", "\n", " ", ""]
        )

        logger.info(f"Text % tabular splitter configured")
        logger.info(f"Vector store directory: {settings.vector_store_dir}")


    async def update_store(self, session_id: str, documents: List[Document]) -> None:
        """Update vector store with new documents"""
        try:
            logger.info(f"Updating vector store for session {session_id} with {len(documents)} documents")
            start_time = time.time()
            
            logger.debug(f"Processing documents for session {session_id}")
            processed_docs = self._process_documents(documents)
            logger.info(f"Documents processed: {len(documents)} input → {len(processed_docs)} chunks")
            
            logger.debug(f"Getting or creating vector store for session {session_id}")
            vector_store = await self._get_or_create_store(session_id, processed_docs)
            
            store_path = self._store_path(session_id)
            logger.debug(f"Saving vector store to {store_path}")
            vector_store.save_local(store_path)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Vector store update completed in {elapsed_time:.2f} seconds for session {session_id}")
        except Exception as e:
            logger.error(f"Vector store update failed for session {session_id}: {e}", exc_info=True)
            raise VectorStoreError(str(e))




    def get_retriever(self, session_id: str) -> Optional[VectorStoreRetriever]:
        """Get retriever for a session"""
        try:
            store_path = self._store_path(session_id)
            logger.info(f"Attempting to load vector store from {store_path} for session {session_id}")
            
            if not os.path.exists(store_path):
                logger.warning(f"Vector store not found for session {session_id} at {store_path}")
                return None
                
            start_time = time.time()
            vector_store = FAISS.load_local(
                store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            elapsed_time = time.time() - start_time
            
            logger.info(f"Vector store loaded successfully in {elapsed_time:.2f} seconds for session {session_id}")
            retriever = vector_store.as_retriever(search_kwargs={"k": 5}) #TODO: We can use other search_types - here the by default similarity search type is chosen
            logger.debug(f"Retriever created with search_kwargs k=5 for session {session_id}")
            # We could implment the multiQueryRetriever for better query optimization and better RAG
            # from langchain.retrievers.multi_query import MultiQueryRetriever
            # from langchain_ollama import ChatOllama
            # llm = ChatOllama(
            #     model=settings.ollama_model,
            #     base_url=settings.ollama_base_url,
            #     temperature=settings.llm_temperature)
            
            # retriver = MultiQueryRetriever.from_llm(
            #     retriever=vector_store.as_retriever(), llm=llm
            # )
            # logger.debug(f"MultiQueryRetriever created for session {session_id}")

                    
            return retriever
        except Exception as e:
            logger.error(f"Error loading vector store for session {session_id}: {e}", exc_info=True)
            return None



    def _process_documents(self, documents: List[Document]) -> List[Document]:
        """Process and split documents"""
        try:
            logger.debug(f"Processing {len(documents)} documents")
            document_metadata = {doc.metadata.get('source', 'unknown'): len(doc.page_content) for doc in documents[:5]}
            logger.debug(f"Sample document sources and sizes: {document_metadata}")
            

            # Group documents by type
            tabular_docs = []
            standard_docs = []
            
            # Sort documents into tabular vs standard
            for doc in documents:
                doc_type = doc.metadata.get('type', '')
                if doc_type in ['csv_row', 'excel_row', 'csv', 'excel']:
                    tabular_docs.append(doc)
                else:
                    standard_docs.append(doc)
            
            logger.info(f"Document types: {len(tabular_docs)} tabular, {len(standard_docs)} standard")
            
            # For tabular data, we'll batch rows together before splitting
            processed_tabular_docs = self._batch_and_split_tabular(tabular_docs)
            
            # For standard documents, use the regular splitter
            start_time = time.time()
            split_standard_docs = self.text_splitter.split_documents(standard_docs) if standard_docs else []
            elapsed_time = time.time() - start_time
            
            logger.info(f"Split {len(standard_docs)} standard documents into {len(split_standard_docs)} chunks in {elapsed_time:.2f} seconds")
            
            # Combine the results
            all_docs = processed_tabular_docs + split_standard_docs
            
            # Add metadata
            for doc in all_docs:
                doc.metadata.update({
                    'chunk_id': str(uuid.uuid4()),
                    'upload_time': time.time(),
                    'recency_flag': True
                })
            
            logger.debug(f"Total chunks after processing: {len(all_docs)}")
            
            return all_docs
        except Exception as e:
            logger.error(f"Document processing failed: {e}", exc_info=True)
            raise

    def _batch_and_split_tabular(self, docs: List[Document]) -> List[Document]:
        """Batch together rows from the same file before splitting"""
        if not docs:
            return []
            
        # Group by source file
        file_groups = {}
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            if source not in file_groups:
                file_groups[source] = []
            file_groups[source].append(doc)
        
        logger.info(f"Batching tabular data from {len(file_groups)} files")
        
        processed_docs = []
        for source, source_docs in file_groups.items():
            # Group by sheet for Excel files
            sheet_groups = {}
            for doc in source_docs:
                sheet = doc.metadata.get('sheet', 'default')
                if sheet not in sheet_groups:
                    sheet_groups[sheet] = []
                sheet_groups[sheet].append(doc)
            
            for sheet, sheet_docs in sheet_groups.items():
                # Sort by row number if available
                sheet_docs.sort(key=lambda x: x.metadata.get('row', 0))
                
                # Batch rows into larger chunks (around 50 rows per chunk)
                batch_size = 50
                for i in range(0, len(sheet_docs), batch_size):
                    batch = sheet_docs[i:i+batch_size]
                    
                    # Create a combined document
                    combined_content = "\n\n".join([doc.page_content for doc in batch])
                    row_range = f"{batch[0].metadata.get('row', 'start')}-{batch[-1].metadata.get('row', 'end')}"
                    
                    batch_doc = Document(
                        page_content=combined_content,
                        metadata={
                            'source': source,
                            'type': 'tabular_batch',
                            'sheet': sheet,
                            'row_range': row_range,
                            'original_rows': len(batch)
                        }
                    )
                    
                    # Now split this larger document if needed
                    if len(combined_content) > 5000:
                        split_docs = self.tabular_splitter.split_documents([batch_doc])
                        processed_docs.extend(split_docs)
                    else:
                        processed_docs.append(batch_doc)
        
        logger.info(f"Created {len(processed_docs)} batched chunks from tabular data")
        return processed_docs
    


    async def _get_or_create_store(self, session_id: str, docs: List[Document]):
        """Get existing store or create new one"""
        try:
            store_path = self._store_path(session_id)
            
            if os.path.exists(store_path):
                logger.info(f"Existing vector store found for session {session_id}")
                start_time = time.time()
                
                vector_store = FAISS.load_local(
                    store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                load_time = time.time() - start_time
                logger.debug(f"Loaded existing vector store in {load_time:.2f} seconds")
                
                # Get existing count if possible
                try:
                    existing_count = len(vector_store.index_to_docstore_id)
                    logger.info(f"Existing vector store has {existing_count} vectors")
                except Exception:
                    logger.debug("Could not determine count of existing vectors")
                
                add_start_time = time.time()
                vector_store.add_documents(docs)
                add_time = time.time() - add_start_time
                
                logger.info(f"Added {len(docs)} new document chunks to existing store in {add_time:.2f} seconds")
                return vector_store
            else:
                logger.info(f"No existing vector store found for session {session_id}, creating new store")
                self._ensure_store_directory(session_id)
                
                start_time = time.time()
                vector_store = FAISS.from_documents(docs, self.embeddings)
                create_time = time.time() - start_time
                
                logger.info(f"Created new vector store with {len(docs)} document chunks in {create_time:.2f} seconds")
                return vector_store
        except Exception as e:
            logger.error(f"Vector store operation failed for session {session_id}: {e}", exc_info=True)
            raise

    def _store_path(self, session_id: str) -> str:
        return f"{settings.vector_store_dir}/{session_id}"

    def _ensure_store_directory(self, session_id: str):
        store_path = self._store_path(session_id)
        logger.debug(f"Ensuring directory exists: {store_path}")
        
        try:
            os.makedirs(store_path, exist_ok=True)
            logger.debug(f"Directory confirmed for session {session_id}: {store_path}")
        except Exception as e:
            logger.error(f"Failed to create directory {store_path}: {e}")
            raise