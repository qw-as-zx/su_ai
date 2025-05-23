# GeoJSON Utilities Documentation
## Purpose
The geojson.py module (app/utils/geojson.py) provides utilities for processing and validating GeoJSON data, focusing on geocoding addresses, verifying coordinates, and visualizing differences between original and corrected coordinates. It integrates external geocoding services and language models to enhance geospatial data accuracy.

## Main Responsibilities
Extract GeoJSON data from markdown text.
Calculate distances between geographic coordinates using the Haversine formula.
Detect country codes from addresses using a language model.
Geocode addresses using the OpenCage Geocoding API.
Verify and correct GeoJSON coordinates based on address information.
Visualize original and corrected GeoJSON coordinates on an interactive map.
Ensure robust logging and error handling for all geospatial operations.

## Class/Function Overviews
**Classes**

`CountryCode (Pydantic BaseModel):`
**Purpose**: Validates and stores a two-letter ISO 3166-1 alpha-2 country code.
Attributes:
code: str - ISO 3166-1 alpha-2 country code.


## Functions

`extract_geojson_from_markdown(text: str) -> Dict[str, Any]:`

**Purpose**: Extracts GeoJSON data from markdown text using regex.
**Key Logic**: Searches for JSON blocks within markdown code fences and parses them into a dictionary.
**Return**: Parsed GeoJSON dictionary.
Exceptions: Raises ValueError for invalid or missing GeoJSON.


haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:

**Purpose**: Calculates the great circle distance between two points on Earth using the Haversine formula.
**Key Logic**: Converts coordinates to radians, applies the formula, and returns distance in kilometers.
**Return**: Distance in kilometers.


detect_country_code_with_ollama(address: str) -> str:

**Purpose**: Uses a language model (Ollama) to detect a two-letter country code from an address.
**Key Logic**: Sends address to Ollama model, validates response with CountryCode, and returns lowercase code.
**Return**: Two-letter country code or "unknown" on failure.
Note: Currently commented out in geocode_address due to potential issues in Linux VM environments.


geocode_address(address: str) -> Optional[List[float]]:

**Purpose**: Geocodes an address using OpenCage Geocoding API.
**Key Logic**: Cleans address input, performs geocoding, and returns [longitude, latitude] or None if unsuccessful.
**Return**: List of [longitude, latitude] or None.
Exceptions: Logs errors and returns None on failure.


verify_and_correct_coordinates(geojson_data: Dict[str, Any], max_distance_km: float = 0.4, delay_between_requests: float = 1.0) -> Dict[str, Any]:

**Purpose**: Verifies GeoJSON coordinates against geocoded addresses and corrects them if they exceed a distance threshold.
**Key Logic**: Iterates through FeatureCollection, geocodes addresses, compares distances, and updates coordinates in-place if necessary.
**Return**: Updated GeoJSON dictionary.
Parameters:
max_distance_km: Maximum allowed distance for coordinate validation (default: 0.4 km).
delay_between_requests: Delay to avoid API rate limits (default: 1.0 second).




visualize_geojson_comparison(original_geojson: Dict[str, Any], corrected_geojson: Dict[str, Any]) -> folium.Map:

**Purpose**: Creates an interactive Folium map to compare original and corrected GeoJSON coordinates.
**Key Logic**: Centers map on coordinate centroid, adds blue markers for original points, red markers for corrected points, and purple lines for corrections.
**Return**: Folium Map object.
Note: Uses [latitude, longitude] format for Folium compatibility.



## Interactions with Other Components

`Configuration (app.utils.config):`
Uses settings.ollama_model for language model configuration in detect_country_code_with_ollama.


`Logging (app.utils.logging):`
Utilizes get_logger to create a module-specific logger for tracking operations and errors.


`External Libraries:`
Folium: Generates interactive maps for visualization.
OpenCage Geocoder: Handles address geocoding via API.
Ollama (langchain_ollama.ChatOllama): Provides country code detection (currently disabled in some contexts).
Pydantic: Validates country codes and API responses.
JSON/Regex: Parses GeoJSON from markdown and validates JSON format.



## Key Implementation Notes

Threading/Asynchronous Operations:
Synchronous implementation; no async operations as geocoding and model calls are blocking.
Uses time.sleep in verify_and_correct_coordinates to manage API rate limits.


**Network Calls:**
Makes HTTP requests to OpenCage Geocoding API in geocode_address.
Interacts with Ollama model via HTTP in detect_country_code_with_ollama (commented out in some cases due to VM compatibility issues).
No explicit timeout handling for OpenCage API; relies on library defaults.


**Error Handling:**
Comprehensive try-except blocks in extract_geojson_from_markdown, geocode_address, and detect_country_code_with_ollama.
Logs warnings for non-critical issues (e.g., no geocoding results) and errors for critical failures.
**Return**s safe defaults (e.g., None, "unknown") on failure to ensure robustness.
Raises ValueError for invalid GeoJSON or JSON parsing errors.


**Performance Considerations:**
Implements delay (delay_between_requests) to prevent OpenCage API rate limit issues.
Processes coordinates in-place to minimize memory usage.
Logs processing statistics (features processed, corrections made) for monitoring.


S**ecurity Notes:**
Sanitizes addresses with regex to remove excessive whitespace and trailing commas.
Uses Pydantic for strict validation of country codes.
Hard-coded OpenCage API key (not ideal; should be moved to configuration).


**Extensibility:**
Supports any GeoJSON FeatureCollection with Point geometries and address properties.
Configurable max_distance_km and delay_between_requests for flexible geocoding.
Modular design allows swapping OpenCage for other geocoding services or Ollama for other LLMs.


**Visualization:**
Uses Folium for browser-compatible interactive maps.
Handles cases with no coordinates by defaulting to a Dubai centroid.
Distinguishes corrected vs. original points with color coding and connecting lines.

----
----

# Redis Client Documentation
## Purpose
The redis_client.py module (app/utils/redis_client.py) provides a centralized interface for interacting with a Redis database to manage session state, chat history, and file metadata in a conversational application. It ensures persistent storage and retrieval of session-related data with expiration policies.

## Main Responsibilities
Initialize and verify Redis connections.
Manage user sessions, including creation, status checking, and activity tracking.
Store and retrieve session-associated file metadata.
Handle chat history persistence with size limits.
Track vector store status for sessions.
Provide utility methods for generic key-value operations.
Ensure robust error handling and logging for all Redis operations.

## Class/Function Overviews
**Classes**
`RedisClient:`
**Purpose**: Encapsulates Redis connection and operations for session, file, and chat history management.
Attributes:
client: redis.Redis - Redis client instance initialized with URL from settings.

**Methods**

init(self):

**Purpose**: Initializes the Redis client and verifies the connection.
**Key Logic:** Creates a Redis connection using redis.Redis.from_url and calls _verify_connection.


`_verify_connection(self):`

**Purpose**: Tests Redis connectivity with a ping.
**Key Logic:** Raises ConnectionError if ping fails.
Exceptions: Logs and raises ConnectionError on failure.


`pipeline(self) -> redis.client.Pipeline:`

**Purpose**: Returns a Redis pipeline object for batch operations.
**Return**: Redis pipeline instance.


`init_session(self, session_id: str) -> None:`

**Purpose**: Initializes a new session with default keys and 24-hour expiration.
**Key Logic:** Sets session status, empty chat history, empty file list, and timestamps.
Exceptions: Logs and raises exceptions on Redis errors.


`add_session_file(self, session_id: str, file_meta: Dict) -> None:`

**Purpose**: Appends file metadata to a session’s file list.
**Key Logic:** Retrieves existing files, appends new metadata, and updates Redis.
Exceptions: Logs errors on Redis or JSON parsing failures.


`get_session(self, session_id: str) -> Optional[str]:`

**Purpose**: Checks if a session exists and returns its status.
**Return**: Session status string or None if not found.


`update_session_activity(self, session_id: str) -> None:`

**Purpose**: Updates the last activity timestamp for a session with 24-hour expiration.
**Key Logic:** Sets last_activity:{session_id} key.


`update_session_files(self, session_id: str, filenames: List[str]) -> None:`

**Purpose**: Updates session file list with new filenames, avoiding duplicates.
**Key Logic:** Retrieves existing files, adds unique filenames, and updates Redis.
Exceptions: Logs errors on Redis or JSON parsing failures.


`get_session_files(self, session_id: str) -> List[Any]:`

**Purpose**: Retrieves the list of files associated with a session.
**Key Logic:** Parses JSON file list from Redis, returns empty list on failure.
**Return**: List of file metadata or empty list.
Exceptions: Logs errors and returns empty list on failure.


`get_chat_history(self, session_id: str) -> List[ChatMessage]:`

**Purpose**: Retrieves chat history for a session.
**Key Logic:** Parses JSON history and converts to ChatMessage objects.
**Return**: List of ChatMessage objects or empty list on failure.
Exceptions: Logs errors and returns empty list on failure.


`save_chat_history(self, session_id: str, history: List[ChatMessage]) -> None:`

**Purpose**: Saves chat history with a size limit and 24-hour expiration.
**Key Logic:** Limits history to 20 messages, serializes to JSON, and stores in Redis.
Exceptions: Logs errors on Redis or JSON serialization failures.


`get_vector_store_status(self, session_id: str) -> bool:`

**Purpose**: Checks if a vector store exists for a session.
**Return**: True if vector store key exists, False otherwise.


`update_vector_store_status(self, session_id: str) -> None:`

**Purpose**: Marks a vector store as existing for a session with 24-hour expiration.
**Key Logic:** Sets vectorstore:{session_id} key.


`set(self, key: str, value: str, ex: int = 86400) -> bool:`

**Purpose**: Sets a key-value pair in Redis with optional expiration.
**Return**: True on success, False on failure.
Parameters:
ex: Expiration time in seconds (default: 86400, 24 hours).




`get(self, key: str) -> Optional[str]:`

**Purpose**: Retrieves a value by key from Redis.
**Return**: Value string or None if key not found.



Interactions with Other Components

Configuration (app.utils.config):
Uses settings.redis_url to initialize the Redis client connection.


Logging (app.utils.logging):
Utilizes get_logger to create a module-specific logger for tracking operations and errors.


Models (app.models.schemas):
Uses ChatMessage schema to validate and deserialize chat history.


External Libraries:
Redis: Provides the core database client for all storage operations.
JSON: Handles serialization/deserialization of chat history and file metadata.



Key Implementation Notes

Threading/Asynchronous Operations:
Synchronous implementation; no async operations as Redis client operations are blocking.
Thread-safe due to Redis client’s internal handling, but no explicit threading mechanisms.


Network Calls:
Connects to Redis via URL specified in settings.redis_url.
Performs network operations for all Redis commands (e.g., set, get, ping).
Relies on Redis client’s default timeout settings; no custom timeout handling.


Error Handling:
Comprehensive try-except blocks in all methods interacting with Redis or JSON.
Logs errors with logger.error for Redis failures or JSON parsing issues.
Returns safe defaults (e.g., empty lists, None) on failure to ensure robustness.
Raises ConnectionError in _verify_connection for failed Redis pings.


Performance Considerations:
Uses 24-hour expiration (86400 seconds) for all keys to manage storage.
Limits chat history to 20 messages to prevent excessive memory usage.
Employs JSON for serialization to maintain compatibility and simplicity.


Security Notes:
Relies on settings.redis_url for secure connection configuration (e.g., SSL, authentication).
Uses Pydantic ChatMessage for structured history validation.
No direct input sanitization; assumes valid session_id and data from callers.


Extensibility:
Generic set and get methods allow flexible key-value storage.
Supports arbitrary file metadata in add_session_file.
Pipeline support enables batch operations for future optimization.


Initialization:
Singleton-like redis_client instance created at module level for application-wide use.
Verifies connection on initialization to ensure Redis availability.





