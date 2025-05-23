import threading
import time
import requests
import logging
from typing import Dict, List, Optional, Set
import psutil
import gc
from datetime import datetime, timedelta
from app.utils.config import settings
from app.utils.logging import get_logger

# Create a logger specific to this module
logger = get_logger("app.services.model_manager")

class ModelState:
    """Tracks the state and usage statistics for a model"""
    def __init__(self, name):
        self.name = name
        self.loaded = False
        self.last_used = None
        self.use_count = 0
        self.loading = False
        self.priority = 1  # Default priority (higher = more important)

class DynamicModelManager:
    def __init__(self, base_url, preload_models=None, interval=1800, 
                 memory_threshold=0.85, max_models=None):
        self.base_url = base_url
        self.interval = interval  # Keep-alive ping interval
        self.memory_threshold = memory_threshold  # Memory usage threshold (0-1)
        self.max_models = max_models  # Maximum number of models to keep loaded
        self.stop_event = threading.Event()
        self.thread = None
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Model state tracking
        self.models: Dict[str, ModelState] = {}
        
        # Initialize with preload models if provided
        if preload_models:
            for model in preload_models:
                self.register_model(model, preload=True)
    
    def register_model(self, model_name: str, preload: bool = False, priority: int = 1) -> None:
        """Register a model with the manager"""
        with self.lock:
            if model_name not in self.models:
                model_state = ModelState(model_name)
                model_state.priority = priority
                self.models[model_name] = model_state
                logger.info(f"Registered model: {model_name} (preload={preload}, priority={priority})")
                
                if preload:
                    # Start a thread to load the model
                    thread = threading.Thread(
                        target=self._load_model_thread, 
                        args=(model_name,)
                    )
                    thread.daemon = True
                    thread.start()
    
    def _load_model_thread(self, model_name: str) -> None:
        """Thread function to load a model"""
        try:
            with self.lock:
                if model_name not in self.models:
                    logger.error(f"Attempted to load unregistered model: {model_name}")
                    return
                    
                model_state = self.models[model_name]
                if model_state.loaded or model_state.loading:
                    return
                
                model_state.loading = True
            
            logger.info(f"Loading model: {model_name}")
            
            # Check if we need to free memory first
            self._check_and_free_memory(model_name)
            
            # Attempt to load the model by sending a simple prompt
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model_name, "prompt": "hello", "stream": False},
                timeout=300  # 5 minutes timeout for initial model loading
            )
            
            with self.lock:
                model_state = self.models[model_name]
                if response.status_code == 200:
                    model_state.loaded = True
                    model_state.last_used = datetime.now()
                    logger.info(f"Successfully loaded model: {model_name}")
                else:
                    logger.error(f"Failed to load model: {model_name}. Status: {response.status_code}")
                
                model_state.loading = False
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            with self.lock:
                if model_name in self.models:
                    self.models[model_name].loading = False
    
    def ensure_model_loaded(self, model_name: str, timeout: int = 300) -> bool:
        """
        Ensures the specified model is loaded. If not currently loaded,
        triggers loading and waits up to timeout seconds.
        
        Returns True if model is loaded successfully, False otherwise.
        """
        start_time = time.time()
        
        # Register if not already registered
        with self.lock:
            if model_name not in self.models:
                self.register_model(model_name)
            
            model_state = self.models[model_name]
            
            # If already loaded, just update stats and return
            if model_state.loaded:
                model_state.last_used = datetime.now()
                model_state.use_count += 1
                return True
            
            # If not yet loading, trigger load
            if not model_state.loading:
                thread = threading.Thread(
                    target=self._load_model_thread, 
                    args=(model_name,)
                )
                thread.daemon = True
                thread.start()
        
        # Wait for model to be loaded
        while time.time() - start_time < timeout:
            with self.lock:
                if model_name in self.models and self.models[model_name].loaded:
                    self.models[model_name].last_used = datetime.now()
                    self.models[model_name].use_count += 1
                    return True
            
            # Wait a bit before checking again
            time.sleep(3)
        
        logger.error(f"Timed out waiting for model {model_name} to load")
        return False
    
    def _check_and_free_memory(self, model_to_load: Optional[str] = None) -> None:
        """
        Check memory usage and unload models if needed to free up memory.
        If model_to_load is provided, ensure we have enough memory to load it.
        """
        mem = psutil.virtual_memory()
        if mem.percent / 100 < self.memory_threshold:
            # Memory usage is below threshold, no need to free
            return
            
        logger.warning(f"Memory usage at {mem.percent}% exceeds threshold. Attempting to free memory.")
        
        # Force garbage collection first
        gc.collect()
        
        # If still above threshold, unload some models
        mem = psutil.virtual_memory()
        if mem.percent / 100 >= self.memory_threshold:
            with self.lock:
                # Get list of loaded models (excluding the one we're trying to load)
                loaded_models = [
                    (name, state) for name, state in self.models.items() 
                    if state.loaded and name != model_to_load
                ]
                
                if not loaded_models:
                    logger.warning("No models available to unload")
                    return
                
                # Sort by priority (ascending) and last used time (ascending)
                # Lower priority models that were used longer ago get unloaded first
                loaded_models.sort(key=lambda x: (x[1].priority, x[1].last_used or datetime.min))
                
                # Unload models until we're under threshold or have unloaded all but one
                for name, _ in loaded_models:
                    self._unload_model(name)
                    
                    # Check if we're now below threshold
                    mem = psutil.virtual_memory()
                    if mem.percent / 100 < self.memory_threshold:
                        break
    
    def _unload_model(self, model_name: str) -> None:
        """Unload a model from memory"""
        # In Ollama, we'll trigger unloading by issuing a specific API call
        # This may vary depending on your LLM service
        try:
            logger.info(f"Unloading model: {model_name}")
            
            # For Ollama, we can attempt to unload by calling an API endpoint
            # Note: Ollama 0.1.17+ supports DELETE /api/delete to unload a model from RAM
            try:
                response = requests.delete(
                    f"{self.base_url}/api/unload",
                    json={"model": model_name}
                )
                if response.status_code in [200, 204]:
                    logger.info(f"Successfully unloaded model: {model_name}")
                else:
                    # If unload endpoint isn't available, we'll rely on Ollama's internal LRU cache
                    logger.warning(f"Model unload API returned status {response.status_code}. Falling back to LRU behavior.")
            except:
                # Ollama versions before 0.1.17 don't have an unload endpoint
                # We'll try alternative approaches
                logger.warning(f"Error calling unload API. Attempting alternative unload approach for {model_name}")
                
                # Force garbage collection
                gc.collect()
                
            # Update model state
            with self.lock:
                if model_name in self.models:
                    self.models[model_name].loaded = False
                
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
    
    def ping_models(self) -> None:
        """Ping loaded models to keep them in memory"""
        while not self.stop_event.is_set():
            try:
                # Get the list of models we want to keep loaded (preloaded models)
                models_to_ping = []
                with self.lock:
                    # Filter for models that are loaded and have high priority
                    models_to_ping = [
                        name for name, state in self.models.items() 
                        if state.loaded and state.priority >= 2  # Only ping high priority models
                    ]
                
                for model_name in models_to_ping:
                    try:
                        logger.info(f"Pinging model {model_name} to keep it alive")
                        response = requests.post(
                            f"{self.base_url}/api/generate",
                            json={"model": model_name, "prompt": "hello", "stream": False},
                            timeout=30
                        )
                        
                        with self.lock:
                            if model_name in self.models:
                                if response.status_code == 200:
                                    logger.info(f"Successfully pinged model: {model_name}")
                                    # Update last_used time but don't increment use_count
                                    # since this is a system ping, not a user request
                                    self.models[model_name].last_used = datetime.now()
                                else:
                                    logger.error(f"Failed to ping model: {model_name}. Status: {response.status_code}")
                                    # Mark as unloaded since the ping failed
                                    self.models[model_name].loaded = False
                                    
                    except Exception as e:
                        logger.error(f"Error pinging model {model_name}: {e}")
                        # Mark as unloaded since the ping failed
                        with self.lock:
                            if model_name in self.models:
                                self.models[model_name].loaded = False
                
                # Wait for the specified interval or until stop is called
                self.stop_event.wait(self.interval)
                
            except Exception as e:
                logger.error(f"Error in ping_models loop: {e}")
                # Continue the loop after a short delay
                self.stop_event.wait(60)
    
    def start(self) -> None:
        """Start the keep-alive thread"""
        if self.thread is None or not self.thread.is_alive():
            self.stop_event.clear()
            self.thread = threading.Thread(target=self.ping_models)
            self.thread.daemon = True
            self.thread.start()
            logger.info("Started dynamic model manager service")
    
    def stop(self) -> None:
        """Stop the keep-alive thread"""
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=5)
            logger.info("Stopped dynamic model manager service")
    
    def get_model_status(self) -> Dict[str, Dict]:
        """Return the current status of all registered models"""
        with self.lock:
            return {
                name: {
                    "loaded": state.loaded,
                    "loading": state.loading,
                    "last_used": state.last_used.isoformat() if state.last_used else None,
                    "use_count": state.use_count,
                    "priority": state.priority
                }
                for name, state in self.models.items()
            }

# Initialize the service with reasonable defaults
model_manager = DynamicModelManager(
    base_url=settings.ollama_base_url,
    preload_models=[settings.ollama_model],  # Preload only the chat model
    interval=3600,  # 1 hour ping interval
    memory_threshold=0.85,  # Free memory if usage exceeds 85%
    max_models=3  # Maximum number of models to keep loaded simultaneously
)