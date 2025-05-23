## 📁 Project Root: `./`

This is the root directory of your AI project. It contains:

* `docker-compose.yml` – Docker Compose configuration for multi-container setups.
* `Dockerfile` – Defines how the application image is built.
* `requirements.txt` – Lists the Python dependencies.
* `app/` – The main application codebase, organized by logical components and responsibilities.

---

## 📁 `app/` – Main Application Package

This is the core application directory. It contains various modules grouped by responsibility:

### ➤ `main.py`

* checkout docs_main.md

### ➤ `preload.py`

* checkout docs_main.md

### ➤ `__init__.py`

* Marks the directory as a Python package.

----
#### 🔄 Interconnections Between Files

| File               | Depends On                       | Role                                  |
| ------------------ | -------------------------------- | ------------------------------------- |
| `main.py`          | `preload.py`, `model_manager.py` | App initialization, startup/shutdown  |
| `preload.py`       | `llm_service`, `model_manager`   | Loads required models before startup  |
| `model_manager.py` | System services, config, logging | Dynamically manages model memory/load |

------


## 📁 `app/routers/` – API Endpoint Definitions

Contains FastAPI routers that expose REST API endpoints:

| File                 | Description                                                                                  |
| -------------------- | -------------------------------------------------------------------------------------------- |
| `audio.py`           | Endpoints for handling audio-related operations (e.g., speech recognition or transcription). |
| `chat_completion.py` | Endpoints for generating responses using LLMs (like chat completion).                        |
| `chat_messages.py`   | Manages chat histories, conversation state, message flow.                                    |
| `files.py`           | Handles file uploads, parsing, and metadata processing.                                      |
| `web_search.py`      | Exposes search functionalities using external search or internal retrieval mechanisms.       |

---

## 📁 `app/models/` – Pydantic Models and Schemas

This folder contains request and response data models used in FastAPI for validation and serialization:

| File         | Description                                                                                                                |
| ------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `schemas.py` | Defines all Pydantic classes such as `ChatRequest`, `WebSearchResponse`, `MapSearchResponse`, etc., used across endpoints. |

---

## 📁 `app/services/` – Business Logic / Backend Services

This folder implements core services that support the endpoints:

| File                     | Description                                                                                                |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `file_processing.py`     | Handles logic for reading, parsing, or extracting data from files.                                         |
| `llm_service.py`         | Loads and manages LLMs (including model downloading, initialization, prompt construction).                 |
| `model_manager.py`       | Possibly a central place to register/load/manage models used across different services.                    |
| `vector_store.py`        | Creates and manages vector embeddings and FAISS vector databases for document search or retrieval.         |
| `web_search_services.py` | Implements the logic behind `web_search.py`, possibly calling external APIs or internal search algorithms. |
| `vision_model_test.png`  | Static image, possibly a placeholder or test asset.                                                        |

---

## 📁 `app/utils/` – Helper Functions and Utilities

This folder contains utility modules to support other components:

| File               | Description                                                                                                |
| ------------------ | ---------------------------------------------------------------------------------------------------------- |
| `config.py`        | Central configuration file for environment variables, model URLs, constants (e.g., Ollama URL, model IDs). |
| `geojson_utils.py` | Validates and processes GeoJSON coordinates returned from searches or user input.                          |
| `logging.py`       | Sets up and configures custom logging across the application.                                              |
| `redis_client.py`  | Handles Redis connections and operations (e.g., retrieving and storing chat history, cache management).    |

---

## 📁 `app/exceptions/` – Custom Error Handling

Contains user-defined exceptions:

| File                   | Description                                                                                                       |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `custom_exceptions.py` | Defines custom exception classes used throughout the app (e.g., `ModelNotFoundException`, `InvalidGeoJSONError`). |

---

## 📁 `__pycache__/`

These are Python cache files generated automatically when modules are imported. They’re not usually included in documentation.

---

## Summary Diagram (Text Tree Recap):

```
ocmsvm_finalv3/
├── app/
│   ├── routers/             # All API endpoints
│   ├── models/              # Pydantic schemas
│   ├── services/            # Backend logic
│   ├── utils/               # Utility functions
│   ├── exceptions/          # Custom exceptions
│   ├── main.py              # FastAPI app entry point
│   └── preload.py           # Pre-initialization logic
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---
