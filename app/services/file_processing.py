
# app/services/file_processing.py
import requests
import base64
import time
import os
import magic
import pandas as pd
import mammoth
import mimetypes
import shutil
import uuid
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, 
    Docx2txtLoader, UnstructuredExcelLoader,
    UnstructuredPowerPointLoader, UnstructuredFileLoader
)
from io import BytesIO

from app.utils.config import settings
from app.utils.redis_client import redis_client
from app.exceptions.custom_exceptions import FileProcessingError

from app.utils.logging import get_logger

# Create a logger specific to this module
logger = get_logger("app.services.file_processing")

mimetypes.init()

class FileProcessor:
    def __init__(self):
        self.supported_types = {
            'text/plain': self.process_text,
            'text/csv': self.process_csv,
            'application/pdf': self.process_pdf,
            'application/CDFV2': self.process_excel,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self.process_word,
            'application/msword': self.process_word,
            'application/vnd.ms-excel': self.process_excel,
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': self.process_excel,
            'application/vnd.ms-powerpoint': self.process_presentation,
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': self.process_presentation,
            'image': self.process_image
        }

    async def process_file(self, file_content: bytes, filename: str, session_id: str) -> List[Document]:
        """Main entry point for file processing"""
        try:
            logger.info(f"Starting processing for {filename}, session_id: {session_id}")
            timestamp = int(time.time())
            mime_type = self._detect_mime_type(file_content, filename)
            logger.info(f"Detected MIME type: {mime_type} for {filename}")
            processor = self._get_processor(mime_type)
            logger.debug(f"Selected processor: {processor.__name__} for {filename}")

            # Create session temp directory
            logger.debug(f"Creating temp directory: {settings.temp_dir}/{session_id}")
            temp_dir = os.path.join(settings.temp_dir, session_id)
            os.makedirs(temp_dir, exist_ok=True)
            logger.info(f"Processed {temp_dir} successfully")
            
            
            return await processor(file_content, filename, temp_dir, timestamp)
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            logger.error(f"Failed to process {filename}: {str(e)}", exc_info=True)
            return self._create_error_document(filename, str(e))

    def _get_processor(self, mime_type: str):
        """Get appropriate processor function"""
        if mime_type.startswith('image/'):
            return self.process_image
        return self.supported_types.get(
            mime_type, 
            self.process_generic
        )

    def _detect_mime_type(self, content: bytes, filename: str) -> str:
        """Enhanced MIME type detection"""
        try:
            mime = magic.from_buffer(content[:2048], mime=True)
            if mime == 'application/octet-stream':
                return mimetypes.guess_type(filename)[0] or mime
            return mime
        except Exception as e:
            logger.warning(f"MIME detection failed for {filename}: {e}")
            return mimetypes.guess_type(filename)[0] or 'application/octet-stream'

    def _create_error_document(self, filename: str, error: str) -> List[Document]:
        return [Document(
            page_content=f"Error processing {filename}",
            metadata={
                "error": error,
                "source": filename,
                "type": "error"
            }
        )]

    async def process_text(self, content: bytes, filename: str, temp_dir: str, timestamp: int) -> List[Document]:
        """Process text files with encoding fallback"""
        documents = []
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1']
            content_text = None
            
            for encoding in encodings:
                try:
                    content_text = content.decode(encoding)
                    logger.debug(f"Decoded {filename} with {encoding}")
                    break
                except UnicodeDecodeError:
                    continue

            if content_text:
                return [Document(
                    page_content=content_text,
                    metadata={
                        "source": filename,
                        "type": "text",
                        "upload_time": timestamp,
                        "encoding": encoding
                    }
                )]
            
            # Fallback to TextLoader
            temp_path = os.path.join(temp_dir, filename)
            with open(temp_path, "wb") as f:
                f.write(content)
                
            loader = TextLoader(temp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "source": filename,
                    "type": "text",
                    "upload_time": timestamp
                })
           
            logger.info(f"Txt processing completed for {filename}, created {len(docs)} documents\n\nDetails : {docs}")
            

            return docs

        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            return self._create_error_document(filename, str(e))

    async def process_pdf(self, content: bytes, filename: str, temp_dir: str, timestamp: int) -> List[Document]:
        """Process PDF files"""
        try:
            logger.info(f"Starting PDF processing for {filename}")
            temp_path = os.path.join(temp_dir, filename)
            logger.debug(f"Writing PDF content to {temp_path}")
            with open(temp_path, "wb") as f:
                f.write(content)

            logger.debug(f"Loading PDF with PyPDFLoader from {temp_path}")
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "source": filename,
                    "type": "pdf",
                    "page_count": len(docs),
                    "upload_time": timestamp
                })
            
            logger.info(f"PDF processing completed for {filename}, created {len(docs)} documents\n\nDetails : {docs}")
            
            return docs
        except Exception as e:
            logger.error(f"PDF processing failed for {filename}: {str(e)}", exc_info=True)
            logger.error(f"PDF processing failed: {e}")
            return self._create_error_document(filename, str(e))

    async def process_word(self, content: bytes, filename: str, temp_dir: str, timestamp: int) -> List[Document]:
        """Process Word documents"""
        try:
            logger.info(f"Starting Word processing for {filename}")
            temp_path = os.path.join(temp_dir, filename)
            with open(temp_path, "wb") as f:
                f.write(content)

            if filename.lower().endswith('.docx'):
                try:
                    loader = Docx2txtLoader(temp_path)
                    docs = loader.load()
                except Exception as e:
                    logger.warning(f"Docx2txt failed: {e}")
                    with open(temp_path, "rb") as docx_file:
                        result = mammoth.extract_raw_text(docx_file)
                        docs = [Document(
                            page_content=result.value,
                            metadata={"source": filename}
                        )]
            else:
                loader = UnstructuredFileLoader(temp_path)
                docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "source": filename,
                    "type": "word",
                    "upload_time": timestamp
                })
            
            logger.debug(f"Updated metadata for {filename}")
           
            logger.info(f"Word processing completed for {filename}, created {len(docs)} documents\n\nDetials: {docs}")
            
            return docs
        except Exception as e:
            logger.error(f"Word processing failed: {e}")
            return self._create_error_document(filename, str(e))

    async def process_excel(self, content: bytes, filename: str, temp_dir: str, timestamp: int) -> List[Document]:
        """Process Excel files"""
        try:
            # Try pandas first
            logger.info(f"Starting Excel processing for {filename}")
            df_dict = pd.read_excel(BytesIO(content), sheet_name=None)
            documents = []
            
            for sheet_name, df in df_dict.items():
                # Add sheet summary
                summary = f"Sheet {sheet_name} has {len(df)} rows with columns: {', '.join(df.columns)}"
                documents.append(Document(
                    page_content=summary,
                    metadata={
                        "source": filename,
                        "type": "excel_summary",
                        "sheet": sheet_name,
                        "upload_time": timestamp
                    }
                ))
                
                # Process rows
                batch_size = 50  # Process in small batches for the vector store service to combine
                for batch_idx in range(0, len(df), batch_size):
                    batch_df = df.iloc[batch_idx:batch_idx+batch_size]
                    
                    # Process batch of rows
                    for idx, row in batch_df.iterrows():
                        doc = Document(
                            page_content="\n".join([f"{col}: {val}" for col, val in row.items()]),
                            metadata={
                                "source": filename,
                                "sheet": sheet_name,
                                "row": idx+1,
                                "type": "excel_row",
                                "batch": batch_idx // batch_size,
                                "upload_time": timestamp
                            }
                        )
                        documents.append(doc)
                
            logger.info(f"Excel processing completed for {filename}, created {len(documents)} documents\n\nDetails : {docs}")
            
            return documents
        except Exception as e:
            logger.warning(f"Pandas Excel processing failed: {e}")
            try:
                logger.info(f"Falling back to UnstructuredExcelLoader for {filename}")
                # Fallback to UnstructuredLoader
                temp_path = os.path.join(temp_dir, filename)
                with open(temp_path, "wb") as f:
                    f.write(content)
                loader = UnstructuredExcelLoader(temp_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata.update({
                        "source": filename,
                        "type": "excel",
                        "upload_time": timestamp
                    })
                logger.info(f"Excel processing completed for {filename}, created {len(docs)} documents\n\nDetails: {docs}")    
                return docs
            except Exception as e:
                logger.error(f"Excel processing failed: {e}")
                return self._create_error_document(filename, str(e))


    async def process_csv(self, content: bytes, filename: str, temp_dir: str, timestamp: int) -> List[Document]:
        """Process CSV files"""
        try:
            # Try pandas first
            logger.info(f"Starting CSV processing for {filename}")
            df = pd.read_csv(BytesIO(content))
            documents = []
            
            # Add summary
            summary = f"CSV has {len(df)} rows with columns: {', '.join(df.columns)}"
            documents.append(Document(
                page_content=summary,
                metadata={
                    "source": filename,
                    "type": "csv_summary",
                    "upload_time": timestamp
                }
            ))
            
            # Process rows in batches
            batch_size = 50  # Process in small batches for the vector store service to combine
            for batch_idx in range(0, len(df), batch_size):
                batch_df = df.iloc[batch_idx:batch_idx+batch_size]
                
                # Process batch of rows
                for idx, row in batch_df.iterrows():
                    doc = Document(
                        page_content="\n".join([f"{col}: {val}" for col, val in row.items()]),
                        metadata={
                            "source": filename,
                            "row": idx+1,
                            "type": "csv_row",
                            "batch": batch_idx // batch_size,
                            "upload_time": timestamp
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"CSV processing completed for {filename}, created {len(documents)} documents")
            return documents
        except Exception as e:
            logger.warning(f"Pandas CSV processing failed: {e}")
            try:
                # Fallback to CSVLoader
                logger.info(f"Falling back to CSVLoader for {filename}")
                temp_path = os.path.join(temp_dir, filename)
                with open(temp_path, "wb") as f:
                    f.write(content)
                loader = CSVLoader(temp_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata.update({
                        "source": filename,
                        "type": "csv",
                        "upload_time": timestamp
                    })
                logger.info(f"CSV processing completed for {filename}, created {len(docs)} documents\nDetails: {docs}")
                return docs
            except Exception as e:
                logger.error(f"CSV processing failed: {e}")
                return self._create_error_document(filename, str(e))
        

    async def process_presentation(self, content: bytes, filename: str, temp_dir: str, timestamp: int) -> List[Document]:
        """Process PowerPoint files"""
        try:
            logger.info(f"Starting PowerPoint processing for {filename}")
            temp_path = os.path.join(temp_dir, filename)
            with open(temp_path, "wb") as f:
                f.write(content)

            loader = UnstructuredPowerPointLoader(temp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "source": filename,
                    "type": "presentation",
                    "upload_time": timestamp
                })

            logger.info(f"Presentation processing completed for {filename}, created {len(docs)} documents\nDetails: {docs}")
            return docs
        except Exception as e:
            logger.error(f"Presentation processing failed: {e}")
            return self._create_error_document(filename, str(e))

    async def process_image(self, content: bytes, filename: str, temp_dir: str, timestamp: int) -> List[Document]:
        """Process image files"""
        try:
            logger.info(f"Starting image processing for {filename}")    
            temp_path = os.path.join(temp_dir, filename)
            with open(temp_path, "wb") as f:
                f.write(content)

            description = await self._analyze_image(temp_path)
            logger.info(f"Image processing completed for {filename}\nDescription Got: {description}")
            return [Document(
                page_content=f"Image: {filename}\nDescription: {description}",
                metadata={
                    "source": filename,
                    "type": "image",
                    "description": description,
                    "upload_time": timestamp
                }
            )]
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return self._create_error_document(filename, str(e))

    async def process_generic(self, content: bytes, filename: str, temp_dir: str, timestamp: int) -> List[Document]:
        """Fallback processor for unknown file types"""
        try:
            # First try text processing
            logger.info(f"Starting generic processing for {filename}")
            text_docs = await self.process_text(content, filename, temp_dir, timestamp)
            if any(doc.page_content.strip() for doc in text_docs):
                return text_docs
            
            # Fallback to unstructured loader
            temp_path = os.path.join(temp_dir, filename)
            with open(temp_path, "wb") as f:
                f.write(content)
                
            loader = UnstructuredFileLoader(temp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "source": filename,
                    "type": "generic",
                    "upload_time": timestamp
                })
            logger.info(f"Generic processing completed for {filename}, created {len(docs)} documents")  
            return docs
        except Exception as e:
            logger.error(f"Generic processing failed: {e}")
            return self._create_error_document(filename, str(e))

    async def _analyze_image(self, image_path: str) -> str:
        """Analyze image using Ollama's vision model"""
        try:
            logger.info(f"Starting image analysis for {image_path}")
            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            response = requests.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.ollama_vision_model,
                    "prompt": "Describe this image in detail", # TODO Improve this prompt to get better results
                    "images": [image_data],
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                # logger.info(f"Image analysis completed for {image_path}\nResponse: {response.json().get('response','No description generated...')}")
                return response.json().get("response", "No description generated")
            return "Image analysis failed"
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return "Image analysis error"
        


## TODO: For the csvs and excels files processing, we can either use 
# 1. Pandas and python-executing tool in langchain to create a simple data analysis chain 
# 2. Or we convert the csv file to sql db and then query it using the sql agent 
# ## Reference: https://python.langchain.com/docs/how_to/sql_csv/ 