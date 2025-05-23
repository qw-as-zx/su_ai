# Suadeo RAG Integration

This document explains how to use the integrated Suadeo RAG (Retrieval-Augmented Generation) system in your OCMSVM API.

## Overview

The Suadeo RAG system provides intelligent search capabilities over your Suadeo data catalog using:
- **Vector embeddings** for semantic search
- **FAISS** for efficient similarity search
- **Ollama** for local LLM inference
- **FastAPI** for REST API endpoints

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Suadeo API    │───▶│  Catalog Parser  │───▶│  RAG Processor  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Vector Search   │◀───│  FAISS Index    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌──────────────────┐
│  Ollama LLM     │───▶│  Generated       │
│  Response       │    │  Answer          │
└─────────────────┘    └──────────────────┘
```

## Quick Start

### 1. Start the Services

```bash
# Start all services
docker-compose up -d

# Setup Ollama models
chmod +x setup_ollama.sh
./setup_ollama.sh
```

### 2. Test the API

```bash
# Check health
curl http://localhost:8000/suadeo/health

# Check system status
curl http://localhost:8000/suadeo/status

# Search example
curl -X POST http://localhost:8000/suadeo/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What datasets are available for HR?",
    "k": 5
  }'
```

### 3. Use the Python Client

```python
python usage_example.py
```

## API Endpoints

### POST `/suadeo/search`

Perform semantic search over your catalog.

**Request:**
```json
{
  "query": "What datasets contain employee information?",
  "k": 5,
  "min_similarity": 0.7
}
```

**Response:**
```json
{
  "answer": "Based on the catalog, there are several datasets containing employee information...",
  "confidence": 0.85,
  "sources": [
    {
      "dataset_name": "SIRH_Employee_Data",
      "content_type": "dataset_overview",
      "similarity": 0.92,
      "catalog": "HR_Catalog"
    }
  ],
  "has_sufficient_context": true,
  "query": "What datasets contain employee information?",
  "total_chunks_found": 3
}
```

### GET `/suadeo/status`

Get system status and health information.

### POST `/suadeo/refresh`

Refresh catalog data and rebuild search index (runs in background).

### GET `/suadeo/health`

Simple health check endpoint.

## Configuration

### Environment Variables

```bash
# Ollama configuration
OLLAMA_BASE_URL=http://localhost:11434

# Suadeo API
SUADEO_API_URL=https://webclient-demo.suadeo.com/apiservice

# RAG configuration
RAG_MIN_SIMILARITY=0.7
RAG_MIN_CONTEXT_CHUNKS=2
```

### Model Configuration

Edit `app/models/suadeo_models.py` to customize:

```python
class SuadeoRAGConfig(BaseModel):
    embedding_model: str = "mxbai-embed-large:latest"
    llm_model: str = "mistral-nemo:12b"
    min_similarity_threshold: float = 0.7
    min_context_chunks: int = 2
```

## Directory Structure

```
app/
├── routers/
│   └── suadeo_router.py          # API endpoints
├── models/
│   └── suadeo_models.py          # Pydantic schemas
├── services/
│   ├── suadeo_rag_service.py     # Main RAG orchestration
│   ├── embeddings_service.py     # Ollama embeddings
│   └── vector_store_service.py   # FAISS vector store
├── utils/
│   ├── catalog_utils.py          # Suadeo API client
│   └── parse_catalogue.py        # Catalog data processing
└── exceptions/
    └── suadeo_exceptions.py      # Custom exceptions
```

## Customization

### Adding New Content Types

Edit `parse_catalogue.py` to handle new data types:

```python
def create_search_entries(dataset: Dict[str, Any], catalog_name: str) -> List[Dict]:
    entries = []
    
    # Add your custom processing logic here
    if 'custom_field' in dataset:
        entries.append({
            "content": f"Custom: {dataset['custom_field']}",
            "content_type": "custom_type",
            "metadata": {...}
        })
    
    return entries
```

### Adjusting Search Behavior

Modify `suadeo_rag_service.py`:

```python
def _assess_context_sufficiency(self, similarities, retrieved_chunks):
    # Customize how context sufficiency is determined
    return your_custom_logic(similarities, retrieved_chunks)
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```bash
   # Check if Ollama is running
   docker ps | grep ollama
   
   # Check Ollama logs
   docker logs ollama
   ```

2. **Models Not Available**
   ```bash
   # Re-run setup script
   ./setup_ollama.sh
   
   # Manually pull models
   docker exec ollama ollama pull mxbai-embed-large:latest
   ```

3. **Index Not Building**
   ```bash
   # Check API logs
   docker logs ocmsvm_api
   
   # Force rebuild
   curl -X POST http://localhost:8000/suadeo/refresh
   ```

4. **Low Search Quality**
   - Lower `min_similarity_threshold` in config
   - Increase `k` parameter in search requests
   - Check if catalog data is being parsed correctly

### Performance Tuning

```python
# For larger datasets, use IVF index
vector_store = FAISSVectorStore(dimension=768, index_type="ivf")

# Adjust batch sizes for embedding generation
embeddings = embed_batch(texts, batch_size=20)  # Increase for faster processing
```

## Monitoring

### Logs

```bash
# View RAG service logs
docker logs -f ocmsvm_api | grep "suadeo"

# View Ollama logs
docker logs -f ollama
```

### Metrics

The `/suadeo/status` endpoint provides:
- Index size (number of vectors)
- Initialization status
- Model information
- System health

## Security Considerations

1. **API Access**: Secure your endpoints with authentication
2. **Ollama Access**: Restrict Ollama to localhost/internal network
3. **Data Privacy**: Ensure catalog data complies with privacy requirements
4. **Resource Limits**: Set appropriate limits for Docker containers

## Next Steps

1. **Authentication**: Add JWT/API key authentication
2. **Caching**: Implement Redis for query caching
3. **Monitoring**: Add Prometheus metrics
4. **Scaling**: Consider horizontal scaling for high load
5. **Advanced RAG**: Implement re-ranking and query expansion