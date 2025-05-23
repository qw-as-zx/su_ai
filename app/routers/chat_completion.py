from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
import os
import asyncio
import json
import time
import requests
from app.models.schemas import ChatRequest, ChatResponse, EchoRequest, ChatMessage
from app.utils.redis_client import redis_client
from app.services.vector_store import VectorStoreService
from app.utils.config import settings
from app.services.llm_service import llm_service

from app.utils.logging import get_logger

# Create a logger specific to this module
logger = get_logger("app.routers.chat_completion")

# router = APIRouter(prefix="/chat", tags=["chat"])
router = APIRouter(prefix="", tags=["chat"])

@router.post("/chat_completion")
async def chat_completion(request: EchoRequest):
    """Handle simple chat completion"""
    try:
        
        async def stream_generator():

            ## Verifying the ollama ...
            # logger.info("Sending to Ollama: %s", request.query[:200])  # Trimmed
            # response = requests.post(
            #     f"{settings.ollama_base_url}/api/generate",
            #     json={"model": settings.ollama_model, "prompt": request.query[:200], "stream": True},
            #     stream=True,
            #     timeout=10  # Add timeout!
            # )
            # logger.info("Ollama response status: %s", response.status_code)
            # if not response.ok:
            #     logger.error("Ollama returned an error: %s", response.text)
            #     response.raise_for_status()
            
            logger.info(f"Sending  Query: `{request.query[:60]}...` to  Ollama Model: {settings.ollama_model}")  # Trimmed

            try:
                logger.info(f"Sending  Query: `{request.query[:60]}...` to  Ollama Model: {settings.ollama_model}")  # Trimmed
                response = requests.post(
                    f"{settings.ollama_base_url}/api/generate",
                    json={
                        "model": settings.ollama_model,
                        "prompt": request.query,
                        "stream": True
                    },
                    stream=True
                )
                
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            yield data.get("response", "")
                        except json.JSONDecodeError:
                            continue
                logger.info("Stream generation completed successfully.")
                
            except Exception as e:
                yield f"Error: {str(e)}"
            logger.info(f"Chat completion Request Response: Code {response}")

        return StreamingResponse(stream_generator(), media_type="text/plain")
    
    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(500, "Completion failed")