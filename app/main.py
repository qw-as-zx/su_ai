from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import chat_completion, files, audio, chat_messages, web_search, suadeo_search
from .utils.config import settings
from fastapi.responses import JSONResponse

from app.utils.logging import get_logger
from app.services.model_manager import model_manager
from app.preload import preload_models

from app.services.rag_services import rag_manager

from app.services.suadeo_rag_service import SuadeoRAGService
from app.models.suadeo_models import SuadeoRAGConfig



# Create a logger specific to this module
logger = get_logger("app.services.main")

app = FastAPI(
    title="Dynamic Multimodal RAG with Suadeo Seach",
    version="1.0.0",
    openapi_url="/openapi.json"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.on_event("startup")
async def startup_event():
    # Start the model management service
    logger.info("Starting application...")
    await preload_models()
    model_manager.start()
    logger.info("Application started - dynamic model manager initialized")

    # Initialize RAG service
    success = await rag_manager.initialize_service(force_rebuild=False)
    if success:
        logger.info("Application started - dynamic model manager and RAG service initialized")
    else:
        logger.warning("Application started - dynamic model manager initialized, RAG service initialization failed")


@app.on_event("shutdown")
async def shutdown_event():
    # Stop the model management service
    model_manager.stop()
    logger.info("Application shutting down - dynamic model manager stopped")

# ✅ Health check endpoint
@app.get("/health")
async def health_check():
    return JSONResponse(status_code=200, content={"status": "ok"})

# ✅ Model status endpoint - NEW!
@app.get("/model-status")
async def model_status():
    """Get the current status of all registered models"""
    status = model_manager.get_model_status()
    return JSONResponse(status_code=200, content=status)


app.include_router(chat_completion.router)
app.include_router(chat_messages.router)
app.include_router(files.router)
app.include_router(audio.router)
# app.include_router(sql_agent.router)
app.include_router(web_search.router)
app.include_router(suadeo_search.router)