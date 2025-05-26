import time
import asyncio
import logging
from typing import Dict, Any
from openai import AsyncOpenAI, OpenAIError
from app.utils.config import settings
from app.utils.redis_client import redis_client
from app.utils.logging import get_logger
from app.utils.geojson_utils import extract_geojson_from_markdown, verify_and_correct_coordinates, visualize_geojson_comparison

# Create a logger specific to this module
logger = get_logger("app.services.web_search_service")


class WebSearchService:
    def __init__(self):
        # OpenAI API key for web search
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def perform_web_search(
        self,
        query: str,
        search_context_size: str = "high", # low, medium, high
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Perform a web search using OpenAI's web search tool with rate limiting.
        """
        # Validate search_context_size
        valid_context_sizes = {"high", "medium", "low"}
        if search_context_size not in valid_context_sizes:
            raise ValueError(f"search_context_size must be one of {valid_context_sizes}")

        # Check rate limit
        rate_limit_key = f"web_search:rate_limit:{user_id}"
        current_time = int(time.time())
        
        try:
            # Increment call count in Redis
            pipe = redis_client.pipeline()
            pipe.incr(rate_limit_key)
            pipe.expire(rate_limit_key, settings.web_search_rate_window)
            call_count, _ = pipe.execute()

            if call_count > settings.web_search_rate_limit:
                logger.warning(f"Rate limit exceeded for user {user_id}: {call_count} calls")
                raise Exception(f"Rate limit exceeded: {settings.web_search_rate_limit} calls per {settings.web_search_rate_window} seconds")

            # Perform API call with retry logic
            for attempt in range(3):
                try:
                    # Configure the web search tool
                    tool = {
                        "type": "web_search_preview",
                        "search_context_size": search_context_size
                    }

                    # Make the API call
                    response = await self.client.responses.create(
                        model="gpt-4.1",
                        tools=[tool],
                        tool_choice={"type": "web_search_preview"},
                        input=query
                    )

                    # Log the raw response for debugging
                    logger.debug(f"Raw OpenAI response: {response}")

                    # Initialize output
                    output = {"text": "", "citations": []}

                    # Process the response.output
                    for item in response.output:
                        if item.type == "message" and item.content:
                            content = item.content[0]
                            if content.type == "output_text":
                                output["text"] = content.text
                                output["citations"] = [
                                    {
                                        "url": ann.url,
                                        "title": ann.title,
                                        "start_index": ann.start_index,
                                        "end_index": ann.end_index
                                    }
                                    for ann in content.annotations
                                    if ann.type == "url_citation"
                                ]
                            break

                    if not output["text"]:
                        raise ValueError("No valid message content found in the response")

                    logger.info(f"Web search successful for query: {query}")
                    return output

                except OpenAIError as e:
                    if "rate_limit" in str(e).lower():
                        logger.warning(f"OpenAI rate limit hit on attempt {attempt + 1} for query: {query}")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        logger.error(f"OpenAI error: {str(e)}")
                        raise
                except Exception as e:
                    logger.error(f"Unexpected error during web search: {str(e)}")
                    raise

            raise Exception("Max retries reached for OpenAI API call")

        except Exception as e:
            logger.error(f"Web search failed: {str(e)}", exc_info=True)
            raise Exception(f"Error during web search: {str(e)}")


    ## V3

    async def perform_map_search(
        self,
        query: str,
        search_context_size: str = "high", 
        user_id: str = "default",
        
    ) -> Dict[str, Any]:
        """
        Perform a map-specific web search and return both parsed GeoJSON data and citations.
        """
        logger.info(f"Performing map search for query: {query}")
        # Validate search_context_size
        valid_context_sizes = {"high", "medium", "low"}
        if search_context_size not in valid_context_sizes:
            raise ValueError(f"search_context_size must be one of {valid_context_sizes}")

        rate_limit_key = f"map_search:rate_limit:{user_id}"
        current_time = int(time.time())

        try:
            pipe = redis_client.pipeline()
            pipe.incr(rate_limit_key)
            pipe.expire(rate_limit_key, settings.web_search_rate_window)
            call_count, _ = pipe.execute()

            if call_count > settings.web_search_rate_limit:
                logger.warning(f"Rate limit exceeded for map search user {user_id}: {call_count} calls")
                raise Exception(f"Rate limit exceeded: {settings.web_search_rate_limit} calls per {settings.web_search_rate_window} seconds")

            for attempt in range(3):
                try:
                    tool = {
                        "type": "web_search_preview",
                        "search_context_size": search_context_size
                    }

                    geojson_prompt = (
                            f"Search for location-specific information for the query: '{query}'. "
                            "Return the results as a valid GeoJSON FeatureCollection. "
                            "Each feature must have a Point geometry with coordinates [longitude, latitude] "
                            "and properties including 'name', 'address', and 'type' (e.g., restaurant, landmark). "
                            "Include only relevant locations. "
                            "Double check the longitude and Latitude, Makesure the coordinates are exactly correct, so they can be put on the map."
                            "Wrap the GeoJSON in triple backticks with a json label (```json\n...\n```). "
                            "If no relevant locations are found, return an empty FeatureCollection. "
                            "Example output:\n"
                            "```json\n"
                            "{\n"
                            "  \"type\": \"FeatureCollection\",\n"
                            "  \"features\": [\n"
                            "    {\n"
                            "      \"type\": \"Feature\",\n"
                            "      \"geometry\": {\"type\": \"Point\", \"coordinates\": [-0.1278, 51.5074]},\n"
                            "      \"properties\": {\"name\": \"Example Place\", \"address\": \"123 London St\", \"type\": \"restaurant\"}\n"
                            "    }\n"
                            "  ]\n"
                            "}\n"
                            "```"
                        )

                    response = await self.client.responses.create(
                        model="gpt-4.1",
                        tools=[tool],
                        tool_choice={"type": "web_search_preview"},
                        instructions="You are a Web Search Assistant who provides location-based GeoJSON search results only.",
                        # input=system_prompt
                        input=geojson_prompt
                    )

                    logger.debug(f"Raw Map Response: {response}")

                    geojson_text = ""
                    citations = []
                    
                    # Extract both the GeoJSON text and citations
                    for item in response.output:
                        if item.type == "message" and item.content:
                            content = item.content[0]
                            if content.type == "output_text":
                                geojson_text = content.text.strip()
                                
                                # Extract citations from annotations
                                citations = [
                                    {
                                        "url": ann.url,
                                        "title": ann.title,
                                        "start_index": ann.start_index,
                                        "end_index": ann.end_index
                                    }
                                    for ann in content.annotations
                                    if ann.type == "url_citation"
                                ]
                                break

                    if not geojson_text:
                        raise ValueError("No GeoJSON content returned.")

                    # geojson_data = extract_geojson_from_markdown(geojson_text)
                    # from verify_geojson_coordinates import verify_and_correct_coordinates
                    # Your GeoJSON data from extract_geojson_from_markdown
                    geojson_data = extract_geojson_from_markdown(geojson_text)
                    geojson_data_ = extract_geojson_from_markdown(geojson_text)
                    # Verify and correct coordinates
                    corrected_geojson = verify_and_correct_coordinates(geojson_data)

                    # Optional: Create a visualization of the changes
                    map_visualization = visualize_geojson_comparison(geojson_data_, corrected_geojson)
                    map_visualization.save('coordinate_corrections.html')
                    
                    return {
                        "geojson": corrected_geojson,
                        "citations": citations
                    }

                except OpenAIError as e:
                    if "rate_limit" in str(e).lower():
                        logger.warning(f"OpenAI map search rate limit hit on attempt {attempt + 1}: {query}")
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.error(f"OpenAI map search error: {str(e)}")
                        raise
                except Exception as e:
                    logger.error(f"Unexpected map search error: {str(e)}")
                    raise

            raise Exception("Max retries reached for OpenAI map search")

        except Exception as e:
            logger.error(f"Map search failed: {str(e)}", exc_info=True)
            raise Exception(f"Error during map search: {str(e)}")