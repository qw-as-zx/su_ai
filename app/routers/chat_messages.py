import requests
import datetime

from langchain_core.messages import HumanMessage

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
import os
import asyncio
import json
import time
from pydantic import BaseModel

from app.models.schemas import ChatRequest, ChatResponse, ChatMessage
from app.utils.redis_client import redis_client
from app.services.vector_store import VectorStoreService
from app.utils.config import settings

from app.services.llm_service import llm_service

from app.utils.logging import get_logger

# Create a logger specific to this module
logger = get_logger("app.routers.chat_messages")


router = APIRouter(prefix="", tags=["chat"])

class RetrievalResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float = 0.0

async def retrieve_context(vector_store: VectorStoreService, session_id: str, query: str) -> List[RetrievalResult]:
    """
    Retrieve relevant context from vector store and format results with metadata
    """
    try:
        logger.info(f"\nRetrieving context for session {session_id}")
        # Check if any documents are uploaded for this session
        session_files = redis_client.get_session_files(session_id) or []
        if not session_files:
            logger.info(f"\nNo documents uploaded for session {session_id}. Skipping retrieval.")
            return []
        
        retriever = vector_store.get_retriever(session_id)
        
        if not retriever:
            logger.warning(f"\nNo retriever available for session {session_id}")
            return []
            
        start_time = time.time()
        # Use proper invoke method instead of deprecated get_relevant_documents
        docs = retriever.invoke(query)
        retrieval_time = time.time() - start_time
        
        logger.info(f"\nRetrieved {len(docs)} documents in {retrieval_time:.2f} seconds")
        
        # Transform documents to RetrievalResult objects
        results = []
        for doc in docs:
            # Calculate recency score based on upload timestamp
            recency_score = 0.0
            if "upload_time" in doc.metadata:
                age_hours = (time.time() - doc.metadata["upload_time"]) / 3600
                recency_score = max(0, 1 - (age_hours / 24))  # Higher score for newer documents
            
            # Create result with metadata
            result = RetrievalResult(
                content=doc.page_content,
                metadata=doc.metadata,
                score=recency_score  # We could enhance this with relevance score from retriever
            )
            results.append(result)
            
        return results
    except Exception as e:
        logger.error(f"Error retrieving context: {e}", exc_info=True)
        return []

def rerank_results(results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
    """
    Rerank results based on relevance and metadata
    Simple implementation - could be enhanced with a proper reranker model
    """
    if not results:
        return []
   
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    query_keywords = set(query_lower.split())

    # Apply basic reranking based on document metadata and content
    for result in results:
        relevance_score = 0.5  # Default score
        
        # Content-based relevance scoring
        content_lower = result.content.lower()

        # Check for exact phrase matches
        if query_lower in content_lower:
            relevance_score += 0.3

         # Check for keyword matches
        matching_keywords = sum(1 for keyword in query_keywords if keyword in content_lower)
        keyword_score = min(0.3, matching_keywords * 0.05)  # Cap at 0.3
        relevance_score += keyword_score

        # File type-specific boosts
        source = result.metadata.get("source", "").lower()
        if source.endswith(".pdf"):
            relevance_score += 0.05
        elif source.endswith((".xlsx", ".xls", ".csv")) and ("data" in query_lower or "spreadsheet" in query_lower):
            relevance_score += 0.15  # Boost Excel/CSV for data-related queries
        elif source.endswith((".jpg", ".png", ".jpeg")) and ("image" in query_lower or "picture" in query_lower):
            relevance_score += 0.15  # Boost images for image-related queries
            
        # Metadata-based boosts    
        # Boost documents with explicit recency flag
        if result.metadata.get("recency_flag", False):
            relevance_score += 0.1

        # Boost documents with matching page numbers if requested
        if "page" in query_lower and "page" in result.metadata:
            query_parts = query_lower.split()
            for i, part in enumerate(query_parts):
                if part == "page" and i+1 < len(query_parts) and query_parts[i+1].isdigit():
                    requested_page = query_parts[i+1]
                    if str(result.metadata["page"]) == requested_page:
                        relevance_score += 0.3
        
        # Update the overall score (combine with existing recency score)
        # Weight relevance more heavily than recency
        result.score = (result.score * 0.3) + (relevance_score * 0.7)
    
    # Sort by combined score (descending)
    results.sort(key=lambda x: x.score, reverse=True)
    return results

def format_context(results: List[RetrievalResult], max_tokens: int = 12000) -> str:
    """
    Format retrieved context into a string, respecting token limits
    """
    if not results:
        # return "No relevant context found."
        return ""
    
    formatted_contexts = []
    total_length = 0
    token_estimate = 0
    
    for i, result in enumerate(results):
        # Estimate tokens (rough approximation: 4 chars ~= 1 token)
        content_length = len(result.content)
        content_tokens = content_length // 4

        # Check if adding this would exceed our token budget
        if token_estimate + content_tokens > max_tokens:
            # If we've already added some context, stop here
            if formatted_contexts:
                break
            # If this is the first context and it's too large, trim it
            trimmed_length = max_tokens * 4
            result.content = result.content[:trimmed_length] + "..."
            
        # Format source info with enhanced metadata
        source = result.metadata.get("source", "Unknown source")
        file_ext = source.split('.')[-1].lower() if '.' in source else ""
        
        # Different formatting based on file type
        metadata_parts = []
        
        # Add page info for PDFs and documents
        if "page" in result.metadata:
            metadata_parts.append(f"page {result.metadata['page']}")
            
        # Add sheet info for Excel files
        if "sheet" in result.metadata:
            metadata_parts.append(f"sheet '{result.metadata['sheet']}'")
            
        # Add cell range for Excel/CSV data
        if "range" in result.metadata:
            metadata_parts.append(f"range {result.metadata['range']}")
            
        # Add timestamp for time-sensitive data
        if "timestamp" in result.metadata and result.metadata["timestamp"]:
            ts = result.metadata["timestamp"]
            if isinstance(ts, (int, float)):
                dt = datetime.datetime.fromtimestamp(ts)
                metadata_parts.append(f"date {dt.strftime('%Y-%m-%d')}")
                
        # Join all metadata parts
        metadata_str = ", ".join(metadata_parts)
        source_info = f"{source}" + (f" ({metadata_str})" if metadata_str else "")
        
        # Create relevance indicator
        relevance = "★★★" if result.score > 0.8 else "★★" if result.score > 0.5 else "★"
        
        # Format the context entry based on file type
        if file_ext in ['csv', 'xlsx', 'xls']:
            context_entry = f"--- From spreadsheet {source_info} (Relevance: {relevance}) ---\n{result.content}\n"
        elif file_ext in ['jpg', 'png', 'jpeg', 'gif', 'bmp']:
            context_entry = f"--- From image {source_info} (Relevance: {relevance}) ---\n{result.content}\n"
        elif file_ext in ['pdf', 'doc', 'docx']:
            context_entry = f"--- From document {source_info} (Relevance: {relevance}) ---\n{result.content}\n"
        else:
            context_entry = f"--- From {source_info} (Relevance: {relevance}) ---\n{result.content}\n"
        
        formatted_contexts.append(context_entry)
        total_length += content_length
        token_estimate += content_tokens
    
    # Join all formatted contexts
    formatted_text = "\n".join(formatted_contexts)
    
    # Add summary of sources
    source_summary = f"[Retrieved {len(formatted_contexts)} relevant passages from {len(set(r.metadata.get('source', '') for r in results[:len(formatted_contexts)]))} sources]"
    
    return source_summary + "\n\n" + formatted_text


@router.post("/messages", response_model=ChatResponse)
async def chat_completion(request: ChatRequest, vector_store: VectorStoreService = Depends()):
    """
    Enhanced RAG-based chat completion endpoint with proper context retrieval,
    reranking, and streaming response
    """
    session_id = request.session_id
    logger.info(f"\nProcessing chat completion for session {session_id}")
    logger.debug(f"\nQuery: {request.query[:100]}...")
    
    try:
        # Get chat history from Redis
        history = redis_client.get_chat_history(session_id)
        logger.info(f"\nRetrieved {len(history)} previous messages from history")
        
        # Add current user message to history
        user_message = ChatMessage(
            role="user",
            content=request.query,
            timestamp=time.time()
        )
        history.append(user_message)
        redis_client.save_chat_history(session_id, history)
        
        async def generate_stream():
            try:
                # Step 1: Retrieve relevant context from vector store
                retrieval_results = await retrieve_context(vector_store, session_id, request.query)
                
                # Step 2: Rerank results based on relevance and metadata
                reranked_results = rerank_results(retrieval_results, request.query)
                
                session_files = redis_client.get_session_files(session_id) or []
                # Extract filenames from metadata or use directly if strings
                filenames = [
                    f.get("filename", f.get("source", f)) if isinstance(f, dict) else f
                    for f in session_files
                ]
                file_types = {f.split('.')[-1].lower() for f in filenames if '.' in f}

                # Step 3: Format context for inclusion in prompt
                context_text = format_context(reranked_results)
                sources = []
                
                # Save source information for later reference
                for result in reranked_results:
                    sources.append({
                        "content": result.content[:8000],  # Truncate long content
                        "metadata": result.metadata,
                        "score": result.score
                    })
                
                # Step 4: Format chat history for context
                # Only include last few messages to avoid context window limits
                formatted_history = []
                for msg in history[-10:]:  # Last 6 messages
                    role = "User" if msg.role == "user" else "Assistant"
                    # Truncate very long messages
                    content = msg.content[:1000] + "..." if len(msg.content) > 1000 else msg.content
                    formatted_history.append(f"{role}: {content}")
                
                history_text = "\n".join(formatted_history)
                
                # Step 5: Construct prompt with system instructions, context, and history
                # Build file-type specific instructions
                file_type_instructions = ""
                if file_types and reranked_results:
                    file_type_instructions="File type specific instructions:\n"
                    
                    # if file_types:
                    #     file_type_instructions = "File type specific instructions:\n"

                    if any(ft in ['xlsx', 'xls', 'csv'] for ft in file_types):
                            file_type_instructions += """
                    - For Excel/CSV data: 
                        * Mention specific cells, ranges, or column headers when referring to data
                        * Provide numerical analysis when appropriate (averages, totals, etc.)
                        * Reference sheet names for Excel files when available
                    """
                    if any(ft in ['pdf', 'doc', 'docx', 'txt', 'rtf'] for ft in file_types):
                            file_type_instructions += """
                    - For text documents: 
                        * Reference specific sections, headings, or page numbers when possible
                        * Quote important passages directly when relevant
                        * Distinguish between document metadata and content
                    """
                        
                    if any(ft in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'] for ft in file_types):
                            file_type_instructions += """
                    - For images: 
                        * Reference visual elements that have been extracted as text
                        * Note that not all visual content may be captured accurately
                        * Be cautious about text extracted from complex images
                    """
                               
                system_prompt = """You are an AI assistant that helps users understand their documents and analyze data.

                Response Guidelines:
                1. Content Focus:
                   - Answer based on user queries
                   - Stay focused on the specific question
                   - If document context is provided, use it to inform your response
                   - For spreadsheet data, provide specific insights
                   - Highlight numerical patterns and trends
                   - Include relevant column names and data points
                
                2. Formatting Rules:
                   - Use # for main topics (Add blank line before and after)
                   - Use ## for subtopics (Add blank line before and after)
                   - Use * for bullet points (Add space after *)
                   - Use proper numbered lists (1., 2., etc.)
                   - Use **bold** for emphasis on key terms
                   - Use `code` for technical terms
                   - Use ```language for code blocks
                
                3. Document References:
                   - Reference uploaded files when relevant
                   - Include specific data points from spreadsheets
                   - Cite the source file and sheet name if applicable
                   
                4. Document References (When provided):
                   - Reference uploaded files when relevant
                   - Include specific data points from spreadsheets
                   - Cite the source file and sheet name if applicable
                   - Format: (Source: filename.ext)

                {file_type_instructions}
                """

                final_prompt = f"""{system_prompt}

                {f"DOCUMENT CONTEXT:\n{context_text}" if context_text else ""}

                CONVERSATION HISTORY:
                {history_text}

                Current question: {request.query}
                
                Provide a helpful, accurate response using the document context when relevant.
                If the document context doesn't contain relevant information, say so clearly.
                """
                
                # Step 6: Get chat model and send request
                logger.info(f"\nSending completion request to model: {settings.ollama_model}")
                chat_model = llm_service.get_chat_model()

                
                if not chat_model:
                    yield "Error: Chat model unavailable"
                    return
                
                # Advanced streaming response handling
                response_text = ""
                current_block = ""
                in_code_block = False
                block_type = None
                
                # Get the streaming response from the model
                # Choose ONE of these approaches:

                # APPROACH 1: Using LangChain's chat model interface (recommended)
                
                messages = [
                    HumanMessage(content=final_prompt)
                ]
                
                # Initialize an empty string to accumulate the response
                full_response = ""
                
                # Process the streaming response
                async for chunk in chat_model.astream(messages):
                    if not chunk:
                        continue
                        
                    # Handle the chunk (assuming chunk is a string)
                    chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    full_response += chunk_content
                    
                    # Handle streaming output formatting
                    for char in chunk_content:
                        if in_code_block:
                            current_block += char
                            if current_block.endswith("```"):
                                yield current_block + "\n\n"
                                current_block = ""
                                in_code_block = False
                                block_type = None
                            continue

                        if not block_type:
                            if current_block.strip().startswith("#"):
                                if response_text:
                                    yield "\n"
                                block_type = "heading"
                            elif current_block.strip().startswith("*") or current_block.strip().startswith("-"):
                                block_type = "list"
                            elif current_block.strip().startswith("```"):
                                if response_text:
                                    yield "\n"
                                in_code_block = True
                                block_type = "code"
                            elif current_block.strip().startswith(">"):
                                block_type = "quote"

                        current_block += char

                        if char == "\n":
                            if block_type == "heading":
                                yield current_block + "\n"
                                current_block = ""
                                block_type = None
                            elif block_type == "list":
                                if current_block.strip():
                                    yield current_block + "\n"
                                current_block = ""
                            elif block_type == "quote":
                                yield current_block + "\n"
                                current_block = ""
                                block_type = None
                            else:
                                if current_block.strip():
                                    yield current_block + "\n"
                                current_block = ""
                                block_type = None

                        if not current_block.strip():
                            block_type = None

                    await asyncio.sleep(0.02)  # Small delay for better streaming
                
                # OR APPROACH 2: Using direct HTTP requests (if you prefer the old way)
                """
                import requests
                import json
                
                response = requests.post(
                    f"{settings.ollama_base_url}/api/generate",
                    json={
                        "model": settings.ollama_model,
                        "prompt": final_prompt,
                        "stream": True
                    },
                    stream=True,
                    timeout=30
                )
                
                if not response.ok:
                    yield f"Error: LLM service returned {response.status_code}"
                    return
                    
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            
                            # [Same streaming formatting logic as above]
                            # ...
                            
                        except json.JSONDecodeError:
                            continue
                """
                
                # Handle any remaining text
                if current_block and current_block.strip():
                    yield current_block
                
                # Save assistant response to history
                logger.debug(f"\nSaving assistant response ({len(full_response)} chars) to history")
                assistant_message = ChatMessage(
                    role="assistant",
                    content=full_response,
                    timestamp=time.time()
                )
                history.append(assistant_message)
                redis_client.save_chat_history(session_id, history)
                
                # Save metadata about the response and sources
                has_sources = len(sources) > 0
                logger.info(f"\nResponse complete. Used {len(sources)} document sources")
                
                redis_client.set(
                    f"last_response_meta:{session_id}",
                    json.dumps({
                        "session_id": session_id,
                        "sources": sources,
                        "has_document_context": has_sources,
                        "retrieval_count": len(retrieval_results),
                        "timestamp": time.time()
                    })
                )
                
            except Exception as e:
                logger.error(f"\nError in generate_stream: {e}", exc_info=True)
                yield f"\nError generating response: {str(e)}"
        
        # Return streaming response
        return StreamingResponse(generate_stream(), media_type="text/plain")
        
    except Exception as e:
        logger.error(f"\nChat completion endpoint error: {e}", exc_info=True)
        raise HTTPException(500, f"Chat processing failed: {str(e)}")

@router.get("/sources/{session_id}")
async def get_last_sources(session_id: str):
    """Retrieve sources used in the last response"""
    try:
        meta_key = f"last_response_meta:{session_id}"
        meta_json = redis_client.get(meta_key)
        
        if not meta_json:
            return {"sources": [], "has_document_context": False}
            
        meta_data = json.loads(meta_json)
        return {
            "sources": meta_data.get("sources", []),
            "has_document_context": meta_data.get("has_document_context", False)
        }
    except Exception as e:
        logger.error(f"Error retrieving sources: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to retrieve sources: {str(e)}")