## 🗂️ Core Application Structure: `main.py`, `preload.py`, `model_manager.py`

---

### 🔹 `main.py` – **FastAPI Application Entrypoint**

**Purpose**:
Orchestrates the FastAPI web server, initializes critical services, and defines core endpoints.

**Key Features**:

* **App Initialization**:

  * Sets up FastAPI with a custom title and OpenAPI doc path.
  * Configures CORS middleware to allow cross-origin requests from all origins.

* **Startup Logic**:

  * Calls `preload_models()` asynchronously to ensure essential models are loaded before serving traffic.
  * Starts the `model_manager`, which keeps models alive in the background.

* **Shutdown Logic**:

  * Gracefully stops the `model_manager` thread on app shutdown.

* **Endpoints**:

  * `/health`: Simple health check to confirm the API is running.
  * `/model-status`: Returns the status of all registered models using `model_manager.get_model_status()`.

* **Router Registration**:

  * Includes multiple routers handling various features (`chat_completion`, `files`, `audio`, `chat_messages`, etc.).

---

### 🔹 `preload.py` – **Model Preloading Logic**

**Purpose**:
Ensures critical models are loaded in memory before the application starts serving user requests.

**Key Actions in `preload_models()`**:

* Logs the beginning of model preloading.
* Uses `llm_service.check_and_init_models()` to ensure models can be initialized (checks Ollama service).
* Registers models with `model_manager` using different **priorities**:

  * Priority `1`: Main chat model (should always be loaded).
  * Priority `2`: Embedding model (frequent usage, preloaded).
  * Priority `0`: Vision model (on-demand loading).
* Triggers actual loading by calling `llm_service.get_chat_model()` and `get_embeddings()`.

**Outcome**:
If successful, key models are in memory and ready. If Ollama is unreachable, a warning is logged.

---

### 🔹 `model_manager.py` – **Dynamic Model Loading & Memory Management**

**Purpose**:
Provides a service to dynamically manage loading, unloading, and keeping AI models alive, with priority-based memory handling.

---

#### 📦 `DynamicModelManager` Class – Core Responsibilities

##### 1. **Model Tracking**:

* Maintains a registry (`self.models`) of models using the `ModelState` class.
* Tracks:

  * If the model is loaded
  * Last usage timestamp
  * Usage count
  * Load status (loading/not)
  * Assigned priority

##### 2. **Model Lifecycle**:

* **`register_model(...)`**:

  * Adds model to the registry and optionally starts loading in a background thread.

* **`_load_model_thread(...)`**:

  * Responsible for contacting the Ollama `/api/generate` endpoint to load a model.
  * On success, updates the model's state.
  * If memory usage is high, may trigger `_check_and_free_memory()` before loading.

* **`ensure_model_loaded(...)`**:

  * Public method to ensure a model is loaded (with timeout).
  * Used by other services that need a model before executing tasks.

* **`_check_and_free_memory(...)`**:

  * Uses `psutil` to monitor memory.
  * If over the defined threshold, unloads lower-priority and less-recently-used models.

* **`_unload_model(...)`**:

  * Tries to unload the model via Ollama's `/api/unload` endpoint (if supported).
  * Fallbacks to garbage collection if the endpoint fails or isn't available.

* **`ping_models(...)`**:

  * Runs in a background thread.
  * Periodically sends a dummy request (e.g., `"hello"`) to high-priority models to keep them in memory.
  * Helps prevent unloading due to inactivity.

* **`start()` / `stop()`**:

  * Starts/stops the ping loop in a background daemon thread.

* **`get_model_status()`**:

  * Returns JSON-style report on all registered models, useful for admin monitoring.

---

## 🔄 Interconnections Between Files

| File               | Depends On                       | Role                                  |
| ------------------ | -------------------------------- | ------------------------------------- |
| `main.py`          | `preload.py`, `model_manager.py` | App initialization, startup/shutdown  |
| `preload.py`       | `llm_service`, `model_manager`   | Loads required models before startup  |
| `model_manager.py` | System services, config, logging | Dynamically manages model memory/load |

---

## 🔑 Key Concepts Used

* **Threading**: For non-blocking model load and keep-alive operations.
* **Memory-aware logic**: Uses `psutil` to ensure efficient use of RAM.
* **Prioritization**: Prevents critical models from being unloaded; allows flexibility.
* **Asynchronous Startup**: Ensures app is ready only after models are loaded.

---
