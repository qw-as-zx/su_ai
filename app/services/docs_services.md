Here’s a **well-structured and technically detailed documentation** for the `file_processing.py` module under your `app/services` directory:

---

# 📄 `file_processing.py`

## Overview

The `file_processing.py` module provides an extensible and asynchronous framework for ingesting, detecting, and extracting structured content from a variety of file formats. It wraps multiple loaders and parsers (primarily from LangChain's ecosystem) and handles metadata enrichment, MIME type identification, fallbacks, and logging. It is optimized for downstream use cases like vector indexing or document Q\&A.

---

## 🧠 Main Responsibilities

* Determine file type using MIME inference and file extension fallbacks.
* Route processing to the correct parser (e.g., PDF, Word, Excel, CSV, images, etc.).
* Normalize all outputs as a list of LangChain `Document` objects.
* Enrich extracted documents with metadata such as source, file type, upload time, row/sheet numbers, etc.
* Provide structured error handling and fallback strategies (e.g., fallback loaders, encoding fallbacks, or raw unstructured parsing).
* Use Ollama vision model for image description (vision-language embedding preparation).

---

## 🔧 Key Class: `FileProcessor`

### `__init__()`

Initializes the MIME type to processor method map, supporting file types like:

* `text/plain`, `text/csv`, `application/pdf`, `application/vnd.*`, `image/*`, etc.

---

## 🔁 Main API

### `async def process_file(file_content: bytes, filename: str, session_id: str) -> List[Document]`

**Primary entry point**. Handles:

* Temp directory creation (`/tmp/<session_id>`)
* MIME detection via `python-magic`
* Dispatching to appropriate processor

Returns:

* `List[Document]` – processed content wrapped with metadata.
* On failure, returns an error Document for observability and UX fallback.

---

## 🔍 MIME Handling

### `_detect_mime_type(content: bytes, filename: str) -> str`

Uses `magic` buffer analysis for primary detection and `mimetypes` as fallback.

---

## 🧰 File Type Processors

Each processor method follows the signature:

```python
async def process_<type>(content: bytes, filename: str, temp_dir: str, timestamp: int) -> List[Document]
```

### ✅ `process_text`

* Tries decoding with `utf-8`, `latin1`, `iso-8859-1`.
* Fallback: LangChain `TextLoader`.
* Adds encoding to metadata.

### ✅ `process_pdf`

* Uses `PyPDFLoader`.
* Adds `page_count` to metadata.

### ✅ `process_word`

* For `.docx`: attempts `Docx2txtLoader`, then `mammoth` as fallback.
* For `.doc`: uses `UnstructuredFileLoader`.

### ✅ `process_excel`

* Primary: `pandas.read_excel` with `sheet_name=None`.
* Sheet summary and row-wise documents (batch size = 50).
* Fallback: `UnstructuredExcelLoader`.

### ✅ `process_csv`

* Primary: `pandas.read_csv`.
* CSV summary + row-wise documents (batch size = 50).
* Fallback: `CSVLoader`.

### ✅ `process_presentation`

* Uses `UnstructuredPowerPointLoader`.

### ✅ `process_image`

* Writes image to disk.
* Uses Ollama's vision model for captioning via POST request.
* Adds image description to `Document`.

### ✅ `process_generic`

* Attempts text decoding.
* Fallback: `UnstructuredFileLoader`.

---

## 📷 Vision Model Integration

### `_analyze_image(image_path: str) -> str`

* Encodes image as base64.
* Sends request to Ollama vision model (`/api/generate`).
* Returns image caption or fallback string.

---

## ❌ Error Handling

### `_create_error_document(filename: str, error: str) -> List[Document]`

Returns a structured error `Document` for fallback UX and pipeline robustness.

---

## 🪵 Logging

* Uses a custom logger via `get_logger`.
* Logs include: MIME detection, loader paths, exceptions, fallbacks, and summaries.

---

## 🔌 External Dependencies

* LangChain loaders: `PyPDFLoader`, `TextLoader`, `CSVLoader`, etc.
* File type libraries: `magic`, `mimetypes`, `mammoth`, `pandas`.
* Ollama (vision model) for image analysis.
* Custom modules: `settings`, `redis_client`, `FileProcessingError`, `get_logger`.

---

## 📌 TODOs / Suggestions (Already present in comments)

* Explore deeper CSV/Excel analysis via:

  1. **Python REPL tools** with LangChain
  2. **SQL-agent-based** querying (e.g., convert CSV to SQLite)
* Improve vision prompt for more descriptive image analysis.

---
---


# 📄 Module: `app.services.llm_services`

## **Purpose**

The `LLMService` class serves as a centralized interface to manage the initialization and usage of Large Language Models (LLMs), including chat models, embedding models, and vision models. It supports both Ollama-based models and a fallback to HuggingFace for embeddings, ensuring resilience and flexibility in different deployment environments.

---

## **Main Responsibilities**

* Detect and verify the availability of the Ollama server.
* Dynamically pull and cache LLM models as needed.
* Initialize and provide access to:

  * Embedding models (Ollama or HuggingFace)
  * Chat-based LLMs (Ollama)
  * Vision models (Ollama)
* Offer health-check style logic to ensure models are ready for use.
* Act as a singleton service to avoid redundant initialization.

---

## 🔧 Class Overview

### `LLMService`

#### **Constructor**

```python
def __init__(self)
```

Initializes the LLM service, verifying Ollama availability and initializing embeddings, chat, and vision models.

---

### **Methods**

#### `_check_ollama_available() -> bool`

Checks whether the Ollama server is reachable via its `/api/tags` endpoint.

#### `_ensure_model_pulled(model_name: str) -> bool`

Validates if a model is locally available via Ollama. If not, attempts to pull it via Ollama’s REST API.

#### `_init_embeddings()`

Initializes the embedding model:

* Uses Ollama if available and specified.
* Falls back to HuggingFace if Ollama is unavailable or not configured.

#### `_init_chat_model() -> Optional[ChatOllama]`

Initializes a chat model through Ollama, ensuring the model is pulled and ready.

#### `_init_vision_model() -> Optional[ChatOllama]`

Initializes a vision-capable chat model using Ollama.

#### `check_and_init_models() -> bool`

Re-checks Ollama availability and re-initializes embeddings, chat, and vision models if necessary.

#### `get_embeddings()`

Returns the active embedding model instance. Reinitializes if missing.

#### `get_chat_model()`

Returns the active chat model instance. Initializes if not present and Ollama is available.

#### `get_vision_model()`

Returns the active vision model instance. Initializes if not present and Ollama is available.

#### `is_ollama_available() -> bool`

Returns the current availability status of the Ollama backend.

---

## 🔗 Interactions with Other Components

| Component                          | Purpose                                                                        |
| ---------------------------------- | ------------------------------------------------------------------------------ |
| `settings` (`app.utils.config`)    | Provides configuration values such as model names, URLs, and temperatures.     |
| `redis_client`                     | Used internally by embedding or LLM libraries, but not directly accessed here. |
| `LLMServiceError`                  | Custom exception raised when initialization or model loading fails.            |
| `get_logger` (`app.utils.logging`) | Provides a structured logger for consistent log output.                        |
| `ChatOllama`, `OllamaEmbeddings`   | Used for chat and embedding models when Ollama is the provider.                |
| `HuggingFaceEmbeddings`            | Used as a fallback for embeddings when Ollama is unavailable.                  |

---

## ⚙️ Key Implementation Notes

* **Network Calls**:

  * Uses HTTP requests to communicate with the Ollama API for model availability and pulling.
  * Includes timeout handling for robustness.

* **Error Handling**:

  * Wraps all critical logic in try-except blocks.
  * Raises `LLMServiceError` on initialization failures.
  * Logs all exceptions with context.

* **Model Lifecycle**:

  * Ensures that each model is pulled before initialization.
  * Reinitializes models only if necessary, minimizing redundant work.

* **Streaming Support**:

  * Chat models are configured to support streaming outputs.

* **Singleton Pattern**:

  * A single instance of `LLMService` is created at module load to ensure reuse across the application.

* **Keep-Alive Configuration**:

  * LLM instances (chat and vision) are configured with `keep_alive=1200` to optimize reuse and avoid reloading.


---
---

# 📄 Module: `app.services.vector_store`

## **Purpose**

The `VectorStoreService` class manages the creation, update, and retrieval of FAISS-based vector stores for document embeddings. It supports session-specific storage and efficient retrieval using LangChain-compatible vector stores.

---

## **Main Responsibilities**

* Initialize and manage embedding models using Ollama or HuggingFace.
* Process and split input documents into manageable chunks for vector indexing.
* Persist and load FAISS vector stores on disk, session-wise.
* Provide a retriever interface for downstream LLM-based semantic search or question answering.
* Handle both standard and tabular document formats with specialized batching and splitting.

---

## 🔧 Class Overview

### `VectorStoreService`

#### **Constructor**

```python
def __init__(self)
```

Initializes embedding models, sets up text splitters for standard and tabular documents, and prepares the vector store configuration based on application settings.

---

### **Public Methods**

#### `async def update_store(session_id: str, documents: List[Document]) -> None`

Processes the input documents and updates or creates a vector store corresponding to the given session ID.

#### `def get_retriever(session_id: str) -> Optional[VectorStoreRetriever]`

Loads the vector store from disk and returns a retriever object configured with similarity search.

---

### **Private Methods**

#### `_process_documents(documents: List[Document]) -> List[Document]`

Splits standard documents and batches tabular data before converting them into chunks suitable for embedding.

#### `_batch_and_split_tabular(docs: List[Document]) -> List[Document]`

Aggregates tabular documents by source and sheet, then groups rows into larger batches before applying splitting logic.

#### `async def _get_or_create_store(session_id: str, docs: List[Document]) -> FAISS`

Loads an existing vector store if it exists or creates a new one using the processed document embeddings.

#### `_store_path(session_id: str) -> str`

Returns the full path for storing the FAISS index for a given session.

#### `_ensure_store_directory(session_id: str)`

Creates the directory on disk if it doesn’t already exist to store the FAISS index.

---

## 🔗 Interactions with Other Components

| Component                        | Purpose                                                                  |
| -------------------------------- | ------------------------------------------------------------------------ |
| `settings` (`app.utils.config`)  | Supplies model type, storage path, and embedding provider.               |
| `llm_service`                    | Provides embedding model instances depending on the configured provider. |
| `FAISS`                          | Used to store, update, and retrieve vector indexes on disk.              |
| `Document`                       | LangChain's document abstraction for text + metadata.                    |
| `RecursiveCharacterTextSplitter` | Splits documents into overlapping text chunks for embeddings.            |
| `VectorStoreRetriever`           | Used for performing vector similarity-based retrieval.                   |
| `VectorStoreError`               | Raised for domain-specific errors related to vector store operations.    |
| `get_logger`                     | Configures module-specific structured logging.                           |

---

## ⚙️ Key Implementation Notes

* **Embedding Initialization**:

  * Supports two providers: Ollama and HuggingFace.
  * Embeddings are retrieved through the shared `llm_service` instance.

* **Text Splitting**:

  * Uses two splitters:

    * `text_splitter` for general-purpose documents.
    * `tabular_splitter` for structured documents like CSV/Excel with higher chunk limits.
  * Tabular documents are batched into \~50 rows before chunking.

* **Vector Store Persistence**:

  * FAISS indexes are saved and loaded from session-specific directories.
  * Uses `FAISS.load_local()` and `FAISS.save_local()` APIs.
  * Enables `allow_dangerous_deserialization=True` for loading.

* **Network Calls**:

  * No direct network calls; relies on embedding providers, which may make outbound API requests internally.

* **Concurrency/Threading**:

  * Public methods like `update_store` and `_get_or_create_store` are `async` to support I/O operations.
  * Thread safety is managed implicitly as FAISS operations are synchronous and isolated per session.

* **Error Handling**:

  * Wrapped in `try-except` blocks with detailed logging.
  * Raises custom `VectorStoreError` where appropriate.
  * Logs exceptions using `exc_info=True` to preserve stack trace.

* **Performance Logging**:

  * Measures and logs time taken for splitting, vector store loading, and updating.
  * Tracks chunk and document counts to aid debugging and tuning.

---
---

# 📄 Module: `app.services.web_search_services`

---

## **Purpose**

`WebSearchService` is responsible for performing web-based and map-specific searches using the OpenAI API. It handles request orchestration, rate limiting, response parsing, and optional post-processing (such as GeoJSON verification and correction for map-based queries).

---

## **Main Responsibilities**

* Enforce per-user rate limiting using Redis.
* Execute web and map search queries using OpenAI’s tool-augmented models.
* Parse and validate response outputs including annotated text and structured geographic data (GeoJSON).
* Apply retry and exponential backoff logic for resilient external API communication.
* Log relevant telemetry and error information for observability and debugging.

---

## 🔧 Class and Function Overviews

##### `class WebSearchService`

Initializes the OpenAI client and manages interaction with the OpenAI API for search tasks.

---

##### `__init__(self)`

Initializes the `AsyncOpenAI` client using credentials from application settings.

---

##### `async def perform_web_search(self, query: str, search_context_size: str = "medium", user_id: str = "default") -> Dict[str, Any]`

Executes a general-purpose web search query using OpenAI’s `web_search_preview` tool.

**Key behaviors:**

* Validates `search_context_size` against allowed values.
* Applies user-specific rate limiting using Redis.
* Sends the query to OpenAI and processes `message` responses.
* Extracts plain text and URL citations from the annotated output.
* Implements exponential backoff with up to 3 retry attempts on transient errors.

**Returns:**
A dictionary containing:

```json
{
  "text": "extracted answer",
  "citations": [ {"url": "...", "title": "...", ...}, ... ]
}
```

---

##### `async def perform_map_search(self, query: str, user_id: str = "default") -> Dict[str, Any]`

Executes a location-specific search and parses the result into a structured `GeoJSON` format.

**Key behaviors:**

* Enforces Redis-based rate limiting.
* Constructs a custom prompt to request GeoJSON-formatted data from OpenAI.
* Parses Markdown-wrapped GeoJSON (` ```json ... ``` `).
* Validates and corrects coordinate accuracy via `verify_and_correct_coordinates`.
* Generates and saves an optional visualization (`coordinate_corrections.html`) showing corrections.

**Returns:**
A dictionary containing:

```json
{
  "geojson": { ... },
  "citations": [ {"url": "...", "title": "...", ...}, ... ]
}
```

---

## 🔗 Interactions with Other Components

* **`AsyncOpenAI` (OpenAI SDK):** Core API for querying models with tool use.
* **`redis_client`:** Manages per-user rate limiting through Redis keys.
* **`settings`:** Provides configuration values such as API keys and rate limits.
* **`extract_geojson_from_markdown`**, **`verify_and_correct_coordinates`**, **`visualize_geojson_comparison`**:
  Handle structured parsing, validation, and visualization of map data.

---

## ⚙️ Key Implementation Notes

* **Rate Limiting:**
  Each user is rate-limited using Redis keys with TTLs (`expire`). This guards against abuse and ensures fair usage across users.

* **Retry Logic:**
  For OpenAI API calls, up to three retries are allowed. Retry delays follow an exponential backoff strategy (`2^attempt` seconds).

* **Error Handling:**
  All external interactions (Redis, OpenAI) are wrapped with `try/except`, and full tracebacks are logged. Errors are raised with context-specific messages.

* **Markdown & GeoJSON Parsing:**
  Responses are expected in a structured Markdown format, requiring careful extraction of valid JSON payloads.

* **File I/O:**
  The corrected coordinate visualization is saved to an HTML file. This operation is synchronous and may block; asynchronous file handling is not implemented.

* **Threading/Concurrency:**
  The class is fully async-compatible but not inherently thread-safe. Should be used within an asyncio-compatible runtime.

---

This service ensures robust and structured integration of OpenAI's web search tooling with enhanced support for geospatial data extraction and validation workflows.


---
---