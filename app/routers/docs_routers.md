### 📁 `app/routers` Module Documentation

---

## Purpose

The `app/routers` module defines API route handlers for core user-facing operations in the application. It leverages **FastAPI** to expose endpoints that handle tasks such as:

* Transcribing audio to text
* Processing and indexing uploaded files
* Performing conversational chat completions via streaming responses

These routes serve as the primary interface between client requests and backend service logic.

---

## Main Responsibilities

* Define RESTful API endpoints grouped by concern (audio, chat, files)
* Validate and handle incoming HTTP requests
* Interface with external services and internal logic (e.g., Whisper, Ollama, Redis, VectorStore)
* Perform request-specific logging and error handling
* Serialize outputs to structured response models

---

## Router Overviews

---

### `audio.py`

#### Endpoint: `POST /audio`

**Purpose:**
Transcribes audio input (any common format) to text using OpenAI's Whisper model.

**Main Components:**

* `whisper.load_model("base")`: Loads Whisper speech-to-text model.
* `convert_to_wav()`: Converts uploaded audio into WAV format using `pydub`.
* `transcribe_audio()`: Reads file, converts format, writes temporary file, and performs transcription.

**Interactions:**

* Uses Whisper for local ML inference.
* Logs input and errors for debugging.
* Relies on temporary disk I/O for model compatibility.

**Key Implementation Notes:**

* Handles format conversion to WAV for compatibility with Whisper.
* Uses `BytesIO` to manipulate audio in memory.
* Raises 400/500 errors via `HTTPException` for invalid input or model failure.
* Uses synchronous file write (`temp_audio.wav`) — not optimal for high concurrency.

---

### `chat_completion.py`

#### Endpoint: `POST /chat_completion`

**Purpose:**
Handles chat completions via a streaming response using the Ollama API.

**Main Components:**

* `chat_completion()`: Main endpoint handler.
* `stream_generator()`: Coroutine that yields text chunks as they are received from Ollama.

**Interactions:**

* Calls the Ollama API at `settings.ollama_base_url` with streaming enabled.
* Parses each line of the stream and yields responses back to the client.
* Logs interactions and stream progress.

**Key Implementation Notes:**

* Uses `StreamingResponse` to handle partial responses in real time.
* Handles JSON decode failures gracefully during stream processing.
* Logs truncated input/output to prevent large logs.
* Network-dependent; does not retry failed requests by default.
* Response format is plain text, not structured JSON.

---

### `files.py`

#### Endpoint: `POST /files`

**Purpose:**
Handles file uploads, extracts document content, and stores them into a vector store for later semantic retrieval.

**Main Components:**

* `upload_files()`: Main upload handler.
* `FileProcessor`: Extracts text from supported file types.
* `VectorStoreService`: Persists documents into a vector store.
* `redis_client`: Tracks session metadata and uploaded files.

**Interactions:**

* Checks/initializes session in Redis.
* Reads file content and delegates processing to `FileProcessor`.
* Updates vector store and Redis with new file metadata.

**Key Implementation Notes:**

* Skips empty files with logging.
* Handles individual file errors gracefully without terminating batch.
* Uses FastAPI’s `Depends()` to inject processor and store services.
* Custom exceptions (`FileProcessingError`, `VectorStoreError`) wrap general exceptions.
* I/O-bound operations (`file.read()`, vector store update) are async-compatible.
* Scalable for multi-file uploads and session-based tracking.

---

## Interactions with Other Components

| Component            | Role                                                           |
| -------------------- | -------------------------------------------------------------- |
| `FastAPI`            | Core web framework for routing and HTTP handling               |
| `whisper`            | Audio transcription engine (used locally)                      |
| `pydub`              | Audio format conversion utility                                |
| `Ollama` API         | External LLM chat completion engine                            |
| `redis_client`       | Tracks session metadata and rate limits                        |
| `VectorStoreService` | Stores processed documents for semantic search                 |
| `FileProcessor`      | Parses and cleans uploaded file content                        |
| `settings`           | Centralized app configuration (API URLs, models, limits, etc.) |
| `Logging`            | Structured module-level log output                             |

---

## Key Implementation Notes

* **Threading & Concurrency**:

  * `audio.py` uses blocking I/O (e.g., file writing).
  * `chat_completion.py` uses streaming and coroutine-based responses.
  * `files.py` is fully async-compatible for file reads and vector store updates.

* **Network Calls**:

  * `chat_completion.py` relies on external HTTP requests to the Ollama API.
  * `files.py` and `audio.py` are local/CPU-bound and do not perform external calls.

* **Error Handling**:

  * Explicit exception handling across all routers.
  * Uses HTTP status codes (`400`, `500`) for consistent API error semantics.
  * Application-specific exceptions wrap low-level issues in `files.py`.

* **Logging**:

  * Each router defines a module-specific logger.
  * Logs inputs (safely truncated), errors, and process steps.
  * Helpful for debugging request behavior and downstream errors.

----
----
# ! Important
### `chat_message.py`
## Purpose
The chat router (app/routers/chat_messages.py) provides the core API endpoints for handling Retrieval-Augmented Generation (RAG) based chat functionality in a document-aware conversational system. It facilitates user interactions by processing queries, retrieving relevant document context, and generating streaming responses using a language model.

## Main Responsibilities

Handle incoming chat requests and return streaming responses.
Retrieve and rank relevant document context from a vector store.
Manage conversation history and session metadata using Redis.
Format responses with proper markdown structure (headings, lists, code blocks).
Provide endpoints to access sources used in responses.
Ensure robust error handling and logging for all operations.

## Class/Function Overviews
### Classes

RetrievalResult (Pydantic BaseModel):
Purpose: Represents a retrieved document with content, metadata, and relevance score.
Attributes:
content: str - Document content.
metadata: Dict[str, Any] - Metadata including source, page, etc.
score: float - Relevance score (default: 0.0).

### Functions

`retrieve_context(vector_store: VectorStoreService, session_id: str, query: str) -> List[RetrievalResult]:`
*Purpose*: Retrieves relevant documents from the vector store for a given session and query.
*Key Logic*: Checks for session files, invokes retriever, calculates recency scores, and returns formatted results.
*Return*: List of RetrievalResult objects.


`rerank_results(results: List[RetrievalResult], query: str) -> List[RetrievalResult]:`

*Purpose*: Reranks retrieved documents based on relevance to the query and metadata.
*Key Logic*: Applies scoring based on keyword matches, file types, and metadata (e.g., page numbers, recency).
*Return*: Sorted list of RetrievalResult objects.


`format_context(results: List[RetrievalResult], max_tokens: int = 12000) -> str:`

*Purpose*: Formats retrieved documents into a string for inclusion in the model prompt.
*Key Logic*: Handles token limits, formats based on file type (PDF, Excel, image), and includes source metadata.
*Return*: Formatted string with source summary and context.


`chat_completion(request: ChatRequest, vector_store: VectorStoreService) -> StreamingResponse:`

*Purpose*: Main POST endpoint (/messages) for processing chat queries and returning streaming responses.
*Key Logic:* 
Retrieves chat history from Redis.
Performs context retrieval and reranking.
Constructs a prompt with system instructions, context, and history.
Streams formatted responses from the language model.
Saves responses and metadata to Redis.
*Return:* StreamingResponse with text/plain media type.


`get_last_sources(session_id: str):`

*Purpose*: GET endpoint (/sources/{session_id}) to retrieve sources used in the last response.
*Key Logic*: Fetches metadata from Redis and returns source information.
*Return*: Dictionary with sources and context usage flag.

**Interactions with Other Components**

Redis Client (app.utils.redis_client):
Stores and retrieves chat history (get_chat_history, save_chat_history).
Manages session files (get_session_files).
Stores response metadata (set, get for last_response_meta).


**Vector Store Service (app.services.vector_store):**
Provides retriever for document context (get_retriever, invoke).


**LLM Service (app.services.llm_service):**
Supplies the chat model for generating responses (get_chat_model, astream).


**Configuration (app.utils.config):**
Provides settings like ollama_model and ollama_base_url.


**Logging (app.utils.logging):**
Logs operations, errors, and performance metrics (get_logger).



## Key Implementation Notes

*Threading/Asynchronous Operations:*
Uses asyncio for asynchronous operations (retrieve_context, chat_completion).
Employs async for for streaming model responses to ensure non-blocking I/O.
Includes asyncio.sleep(0.02) in streaming to manage response pacing.

*Network Calls:*
Relies on requests for HTTP-based model interaction (commented alternative approach).
Uses LangChain's astream for primary model communication, reducing direct network calls.
Redis interactions are network-based but abstracted via redis_client.

*Error Handling:*
Comprehensive try-except blocks in all major functions.
Logs errors with stack traces using logger.error with exc_info=True.
Returns HTTP exceptions (500) for critical failures with descriptive messages.
Gracefully handles empty results or unavailable services (e.g., no retriever, no model).


*Performance Considerations:*
Limits history to last 10 messages to manage context window.
Truncates long content (e.g., sources to 8000 chars, messages to 1000 chars).
Caps token usage in format_context (default: 12000 tokens).
Tracks retrieval time for performance monitoring.


*Security Notes:*
Validates session IDs and query content via Pydantic models (ChatRequest).
Sanitizes file extensions and metadata for safe formatting.
No direct file I/O; relies on vector store for document access.


*Extensibility:*
File-type specific instructions in chat_completion support multiple formats (PDF, Excel, images).
Modular reranking allows for future integration of advanced ranking models.
Streaming response formatting supports markdown elements (headings, lists, code blocks).



