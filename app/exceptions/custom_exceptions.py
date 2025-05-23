from fastapi import HTTPException

class FileProcessingError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=500,
            detail=f"File processing error: {detail}"
        )

class VectorStoreError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=500,
            detail=f"Vector store error: {detail}"
        )

class LLMServiceError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=503,
            detail=f"LLM service error: {detail}"
        )