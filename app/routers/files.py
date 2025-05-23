from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import List
from app.models.schemas import FileUploadResponse
from app.services.file_processing import FileProcessor
from app.services.vector_store import VectorStoreService
from app.utils.redis_client import redis_client
from app.exceptions.custom_exceptions import FileProcessingError, VectorStoreError

from app.utils.logging import get_logger

# Create a logger specific to this module
logger = get_logger("app.routers.files")


router = APIRouter(prefix="/files", tags=["files"])
# logger = logging.getLogger(__name__)

@router.post("", response_model=FileUploadResponse)
async def upload_files(
    session_id: str,
    files: List[UploadFile] = File(...),
    processor: FileProcessor = Depends(),
    vector_store: VectorStoreService = Depends()
):
    """Handle file uploads and processing"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Initialize session
        if not redis_client.get_session(session_id):
            redis_client.init_session(session_id)
        
        processed_docs = []
        for file in files:
            try:
                content = await file.read()
                if len(content) == 0:
                    logger.warning(f"Empty file received: {file.filename}")
                    continue
                
                docs = await processor.process_file(content, file.filename,session_id)
                processed_docs.extend(docs)
                
            except Exception as e:
                logger.error(f"Failed to process {file.filename}: {e}")
                continue
        
        if processed_docs:
            try:
                await vector_store.update_store(session_id, processed_docs)
                redis_client.update_session_files(session_id, [f.filename for f in files])
            except Exception as e:
                logger.error(f"Vector store update failed: {e}")
                raise VectorStoreError(str(e))
        
        return FileUploadResponse(
            message=f"Processed {len(files)} files with {len(processed_docs)} documents"
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise FileProcessingError(str(e))