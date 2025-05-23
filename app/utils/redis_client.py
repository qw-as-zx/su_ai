import json
import time
import redis
from typing import Any, List, Dict, Optional
from app.models.schemas import ChatMessage
from app.utils.config import settings
from app.utils.logging import get_logger

# Create a logger specific to this module
logger = get_logger("app.utils.redis_client")

class RedisClient:
    def __init__(self):
        self.client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self._verify_connection()

    def _verify_connection(self):
        if not self.client.ping():
            logger.error("Redis connection failed")
            raise ConnectionError("Could not connect to Redis")
        
    def pipeline(self):
        """Return a Redis pipeline object."""
        return self.client.pipeline()
    
    # Session management
    def init_session(self, session_id: str) -> None:
        """Initialize a new session with default values"""
        try:
            self.client.set(f"session:{session_id}", "active", ex=86400)  # 24h expiration
            self.set(f"chat_history:{session_id}", json.dumps([]))
            self.set(f"files:{session_id}", json.dumps([]))
            self.set(f"session_created:{session_id}", time.time())
            self.set(f"last_activity:{session_id}", time.time())
        except Exception as e:
            logger.error(f"Session initialization failed: {e}")
            raise

    # File management
    def add_session_file(self, session_id: str, file_meta: Dict) -> None:
        """Add file metadata to session"""
        try:
            files = json.loads(self.client.get(f"files:{session_id}") or "[]")
            files.append(file_meta)
            self.client.set(f"files:{session_id}", json.dumps(files))
        except Exception as e:
            logger.error(f"Failed to update session files: {e}")


    def get_session(self, session_id: str) -> Optional[str]:
        """Check if session exists and return its status"""
        return self.client.get(f"session:{session_id}")
    

    def update_session_activity(self, session_id: str) -> None:
        """Update last activity timestamp"""
        self.client.set(f"last_activity:{session_id}", time.time(), ex=86400)

    def update_session_files(self, session_id: str, filenames: List[str]) -> None:
        """Update session files with new filenames"""
        try:
            files = json.loads(self.client.get(f"files:{session_id}") or "[]")
            for filename in filenames:
                if filename not in files:
                    files.append(filename)
            self.client.set(f"files:{session_id}", json.dumps(files))
        except Exception as e:
            logger.error(f"Failed to update session files: {e}")

    def get_session_files(self, session_id: str) -> List[Any]:
            """Retrieve the list of files associated with the session"""
            try:
                files_json = self.client.get(f"files:{session_id}") or "[]"
                files = json.loads(files_json)
                return files
            except Exception as e:
                logger.error(f"Failed to retrieve session files: {e}")
                return []
            

    # Chat history management
    def get_chat_history(self, session_id: str) -> List[ChatMessage]:
        """Retrieve chat history"""
        try:
            history = json.loads(self.client.get(f"chat_history:{session_id}") or "[]")
            return [ChatMessage(**msg) for msg in history]
        except Exception as e:
            logger.error(f"Failed to get chat history: {e}")
            return []

    def save_chat_history(self, session_id: str, history: List[ChatMessage]) -> None:
        """Persist chat history with size limit"""
        try:
            if len(history) > 20:
                history = history[-20:]
            self.client.set(
                f"chat_history:{session_id}",
                json.dumps([msg.dict() for msg in history]),
                ex=86400
            )
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")

    # Vector store management
    def get_vector_store_status(self, session_id: str) -> bool:
        """Check if vector store exists for session"""
        return self.client.exists(f"vectorstore:{session_id}") == 1

    def update_vector_store_status(self, session_id: str) -> None:
        """Mark vector store as existing"""
        self.client.set(f"vectorstore:{session_id}", "true", ex=86400)

    # Utility methods
    def set(self, key: str, value: str, ex: int = 86400) -> bool:
        return self.client.set(key, value, ex=ex)

    def get(self, key: str) -> Optional[str]:
        return self.client.get(key)

redis_client = RedisClient()