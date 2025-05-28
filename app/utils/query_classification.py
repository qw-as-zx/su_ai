## V4 with Logs

from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import logging
from app.models.suadeo_models import QueryClassification, SuadeoSearchResponse, RelevanceClassification, SuadeoSearchRequest, SearchSource
from app.utils.config import settings
from app.utils.logging import get_logger
from app.exceptions.suadeo_exceptions import IndexNotInitializedError, OllamaConnectionError, SuadeoRAGException
from fastapi import HTTPException, Depends
import json

# Create a logger specific to this module
logger = get_logger("app.routers.suadeo_search_query_handle")

def get_ollama_client(base_url: str =  settings.ollama_base_url):
    """
    Get ChatOllama client with optional base_url
    """
    logger.debug(f"Creating Ollama client with base_url: {base_url}, model: {settings.ollama_greeting_model}")
    
    if base_url:
        client = ChatOllama(
            model=settings.ollama_greeting_model,
            base_url=base_url,
            temperature=0.4
        )
    else:
        client = ChatOllama(
            model=settings.ollama_greeting_model
        )
    
    logger.debug("Ollama client created successfully")
    return client

async def detect_language(query: str, base_url: str =  settings.ollama_base_url) -> str:
    """
    Detect the language of the user's query
    """
    logger.info(f"Starting language detection for query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
    
    try:
        llm = get_ollama_client(base_url)
        
        messages = [
            SystemMessage(content='''Detect the language of the given text. Return only the language name in English (e.g., "English", "Spanish", "French", "German", "Arabic", "Chinese", etc.).
                    
                    If the text contains mixed languages, return the dominant language.
                    If the text is very short or unclear, return "English" as default.'''),
            HumanMessage(content=f'Detect the language of this text: "{query}"')
        ]
        
        logger.debug("Sending language detection request to Ollama")
        response = llm.invoke(messages)
        detected_language = response.content.strip()
        
        logger.info(f"Language detection successful - Detected language: {detected_language}")
        return detected_language
        
    except Exception as e:
        logger.error(f"Error detecting language for query '{query[:30]}...': {str(e)}")
        logger.warning("Falling back to English as default language")
        return "English"  # Fallback to English

async def create_optimized_english_search_query(original_query: str, detected_language: str, base_url: str =  settings.ollama_base_url) -> str:
    """
    Create an optimized English search query for FAISS vector database search.
    This function translates non-English queries and optimizes them for semantic search.
    """
    logger.info(f"Creating optimized search query - Original: '{original_query}', Language: {detected_language}")
    
    try:
        llm = get_ollama_client(base_url)
        
        # If already in English, just optimize for vector search
        if detected_language.lower() == "english":
            logger.debug("Query is in English, optimizing for vector search")
            
            messages = [
                SystemMessage(content='''You are an expert at optimizing search queries for vector database semantic search. 
                        
                        Transform the user's query into an optimized version that will work better with semantic search/FAISS vector databases.
                        
                        Guidelines:
                        - Keep the core meaning intact
                        - Use clear, specific terminology
                        - Remove unnecessary words/filler
                        - Use data catalog terminology when appropriate
                        - Focus on key concepts and entities
                        - Make it simple, without the words like dashboard
                        - Make it more declarative rather than conversational
                        
                        Examples:
                        - "Can you show me info about customer data?" → "customer data information metadata"
                        - "What datasets do we have for sales?" → "sales datasets available"
                        - "I need to find tables related to HR" → "HR tables database schema"
                        
                        Return only the optimized query, nothing else.'''),
                HumanMessage(content=f'Optimize this query for vector search: "{original_query}"')
            ]
            
            logger.debug("Sending optimization request to Ollama")
            response = llm.invoke(messages)
            optimized_query = response.content.strip()
            
            logger.info(f"Query optimization successful - Optimized: '{optimized_query}'")
            return optimized_query
            
        else:
            # Translate and optimize for non-English queries
            logger.debug(f"Query is in {detected_language}, translating and optimizing")
            
            messages = [
                SystemMessage(content=f'''You are an expert translator and search query optimizer. The user has provided a query in {detected_language}.
                        
                        Your task is to:
                        1. Translate the query from {detected_language} to English
                        2. Optimize it for semantic search/FAISS vector database
                        
                        Optimization guidelines:
                        - Keep the core meaning intact
                        - Use clear, specific data catalog terminology
                        - Remove unnecessary words/filler
                        - Focus on key concepts and entities
                        - Make it more declarative rather than conversational
                        - Use terms like: dataset, table, schema, metadata, database, catalog, etc. when relevant
                        
                        Examples of good optimized queries:
                        - "customer data information metadata"
                        - "sales datasets available"
                        - "HR employee database tables"
                        - "financial reports data sources"
                        
                        Return only the optimized English query, nothing else.'''),
                HumanMessage(content=f'Translate and optimize this {detected_language} query for English vector search: "{original_query}"')
            ]
            
            logger.debug("Sending translation and optimization request to Ollama")
            response = llm.invoke(messages)
            optimized_query = response.content.strip()
            
            logger.info(f"Translation and optimization successful - From '{original_query}' ({detected_language}) to '{optimized_query}' (English)")
            return optimized_query
            
    except Exception as e:
        logger.error(f"Error creating optimized search query for '{original_query}': {str(e)}")
        logger.warning(f"Falling back to original query: '{original_query}'")
        # Fallback: return original query
        return original_query
    

async def generate_multilingual_rag_response(
    original_query: str, 
    rag_results: dict, 
    detected_language: str,
    optimized_query: str,
    base_url: str = settings.ollama_base_url
) -> str:
    """
    Generate a response in the original language using RAG results from English search
    """
    logger.info(f"Generating multilingual RAG response - Language: {detected_language}, Sources count: {len(rag_results.get('sources', []))}")
    
    try:
        # If the query was in English, return the original RAG response
        if detected_language.lower() == "english":
            logger.debug("Language is English, returning original RAG response")
            response = rag_results.get('answer', 'No answer available')
            logger.info(f"English RAG response generated, length: {len(response)} characters")
            return response
        
        logger.debug(f"Generating response in {detected_language} using RAG results")
        llm = get_ollama_client(base_url)
        
        # For non-English queries, generate response in original language
        sources_info = ""
        if rag_results.get('sources'):
            sources_count = len(rag_results['sources'])
            logger.debug(f"Processing {sources_count} sources for response context")
            
            sources_info = "Based on the following data sources:\n"
            for i, source in enumerate(rag_results['sources'][:], 1):  # Limit to top 3 sources
                dataset_name = source.get('dataset_name', 'Unknown')
                content_type = source.get('content_type', 'Unknown')
                sources_info += f"{i}. {dataset_name} - {content_type}\n"
                logger.debug(f"Source {i}: {dataset_name} ({content_type})")
        logger.info(f"Source Info: {sources_info}\n\n")
        messages = [
            SystemMessage(content=f'''You are a helpful data catalog assistant. The user asked a question in {detected_language}, and you need to provide a comprehensive answer in {detected_language}.
                    
                    You have search results from a data catalog that will help you answer their question. Use this information to provide a helpful, accurate response.
                    
                    Guidelines:
                    - Respond entirely in {detected_language}
                    - Use the search results to provide specific, relevant information
                    - If the search results don't contain enough information, say so honestly
                    - Maintain a helpful and professional tone
                    - Reference specific datasets or sources when mentioned in the results
                    - Keep the response focused on data catalog information
                    
                    Original user query in {detected_language}: "{original_query}"
                    Search query used: "{optimized_query}"'''),
            HumanMessage(content=f'''Based on these search results from the data catalog, please answer the user's question in {detected_language}:

                        Search Results:
                        Answer: {rag_results.get('answer', 'No specific answer found')}

                        {sources_info}

                        Additional Context:
                        - Confidence: {rag_results.get('confidence', 0):.2f}
                        - Total sources found: {len(rag_results.get('sources', []))}
                        - Has sufficient context: {rag_results.get('has_sufficient_context', False)}

                        Please provide a comprehensive answer in {detected_language}.''')
        ]
        
        logger.debug(f"Sending multilingual response generation request to Ollama")
        response = llm.invoke(messages)
        multilingual_response = response.content.strip()
        
        logger.info(f"Multilingual response generated successfully in {detected_language}, length: {len(multilingual_response)} characters")
        return multilingual_response
        
    except Exception as e:
        logger.error(f"Error generating multilingual response in {detected_language}: {str(e)}")
        logger.warning("Falling back to original RAG response")
        # Fallback to original RAG response
        return rag_results.get('answer', 'I apologize, but I encountered an error generating the response.')

# Your existing classification functions (updated to use ChatOllama)
async def classify_query(query: str, base_url: str =  settings.ollama_base_url) -> QueryClassification:
    """
    Classify if the query is a greeting using Ollama with structured output
    """
    logger.info(f"Classifying query as greeting or not: '{query[:50]}{'...' if len(query) > 50 else ''}'")
    
    try:
        llm = get_ollama_client(base_url)
        
        messages = [
            SystemMessage(content='''You are a query classifier. Analyze the user's input and determine if it's a greeting.
                    
                    Consider these as greetings:
                    - Basic greetings: hi, hello, hey, good morning, good afternoon, good evening, good night
                    - Multilingual greetings: bonjour, hola, ciao, guten tag, konnichiwa, namaste, salaam, etc.
                    - Informal greetings: what's up, how's it going, sup, yo
                    - Polite introductions: good day, greetings
                    
                    NOT greetings:
                    - Questions about products, services, or information
                    - Requests for help or assistance (unless purely greeting)
                    - Commands or instructions
                    - Any substantive query mixed with greeting
                    
                    Provide confidence as a float between 0.0 and 1.0
                    
                    Respond with a valid JSON object matching this schema:
                    {
                        "is_greeting": boolean,
                        "confidence": float,
                        "classification_reason": string
                    }'''),
            HumanMessage(content=f'Classify this query: "{query}"')
        ]
        
        logger.debug("Sending greeting classification request to Ollama")
        response = llm.invoke(messages)
        
        # Parse the JSON response
        try:
            logger.debug(f"Parsing JSON response: {response.content}")
            classification_data = json.loads(response.content)
            classification = QueryClassification(**classification_data)
            
            logger.info(f"Query classification successful - Is greeting: {classification.is_greeting}, Confidence: {classification.confidence:.2f}")
            logger.debug(f"Classification reason: {classification.classification_reason}")
            
        except (json.JSONDecodeError, ValueError) as parse_error:
            logger.error(f"Failed to parse JSON response from greeting classification: {parse_error}")
            logger.debug(f"Raw response content: {response.content}")
            # Fallback if JSON parsing fails
            raise Exception("Failed to parse JSON response")
            
        return classification
        
    except Exception as e:
        logger.error(f"Error classifying query as greeting: {str(e)}")
        logger.warning("Falling back to keyword-based greeting detection")
        
        # Fallback to simple keyword detection
        greeting_keywords = [
            'hi', 'hello', 'hey', 'bonjour', 'hola', 'ciao', 'good morning', 
            'good afternoon', 'good evening', 'good night', 'greetings',
            'what\'s up', 'how\'s it going', 'sup', 'yo', 'guten tag',
            'konnichiwa', 'namaste', 'salaam', 'good day'
        ]
        
        query_lower = query.lower().strip()
        is_greeting = any(keyword in query_lower for keyword in greeting_keywords)
        
        logger.info(f"Fallback greeting classification - Is greeting: {is_greeting}")
        
        return QueryClassification(
            is_greeting=is_greeting,
            confidence=0.8 if is_greeting else 0.2,
            classification_reason="Fallback keyword detection"
        )

async def classify_relevance(query: str, base_url: str =  settings.ollama_base_url) -> RelevanceClassification:
    """
    Classify if the query is relevant to Suadeo catalog using Ollama with structured output
    """
    logger.info(f"Classifying query relevance to Suadeo catalog: '{query[:50]}{'...' if len(query) > 50 else ''}'")
    
    try:
        llm = get_ollama_client(base_url)
        
        messages = [
            SystemMessage(content='''You are a relevance classifier for Suadeo data catalog queries. 
                    
                    Suadeo is a data catalog platform that helps organizations discover, understand, and manage their data assets.
                    
                    RELEVANT queries are about:
                    - Data discovery and search
                    - Dataset information, metadata, schemas
                    - Suadeo Catalog
                    - Personal questions any name, or user
                    - Questions about any dataset name, catalog entity like dataset_ids, domains, owner etc 
                    
                    IRRELEVANT queries are about:
                    - General conversation not related to data
                    - Weather, news, entertainment
                    - Programming help unrelated to data
                    - Mathematical calculations (unless data-related)
                    - General knowledge questions
                    - Product recommendations (non-data related)
                    - Travel, food, lifestyle topics
                    - Technical support for non-data tools
                    - Creative writing or storytelling
                    
                    Edge cases - Consider context:
                    - "API" could be relevant if asking about data APIs
                    - "Database" is usually relevant
                    
                    If unsure, lean towards RELEVANT to avoid blocking legitimate data queries.
                    
                    Provide confidence as a float between 0.0 and 1.0. 
                    If not relevant, suggest 2-3 relevant topics the user could ask about instead.
                    
                    Respond with a valid JSON object matching this schema:
                    {
                        "is_relevant": boolean,
                        "confidence": float,
                        "classification_reason": string,
                        "suggested_topics": array of strings (optional, only if not relevant)
                    }'''),
            HumanMessage(content=f'Classify the relevance of this query to Suadeo data catalog: "{query}"')
        ]
        
        logger.debug("Sending relevance classification request to Ollama")
        response = llm.invoke(messages)
        
        # Parse the JSON response
        try:
            logger.debug(f"Parsing JSON response: {response.content}")
            classification_data = json.loads(response.content)
            classification = RelevanceClassification(**classification_data)
            
            logger.info(f"Relevance classification successful - Is relevant: {classification.is_relevant}, Confidence: {classification.confidence:.2f}")
            logger.debug(f"Classification reason: {classification.classification_reason}")
            
            if not classification.is_relevant and classification.suggested_topics:
                logger.debug(f"Suggested topics for irrelevant query: {classification.suggested_topics}")
            
        except (json.JSONDecodeError, ValueError) as parse_error:
            logger.error(f"Failed to parse JSON response from relevance classification: {parse_error}")
            logger.debug(f"Raw response content: {response.content}")
            # Fallback if JSON parsing fails
            raise Exception("Failed to parse JSON response")
            
        return classification
        
    except Exception as e:
        logger.error(f"Error classifying query relevance: {str(e)}")
        logger.warning("Falling back to keyword-based relevance detection")
        
        # Fallback - be permissive to avoid blocking legitimate queries
        data_keywords = [
            'data', 'dataset', 'table', 'database', 'sql', 'query', 'schema',
            'api', 'endpoint', 'catalog', 'metadata', 'column', 'field',
            'source', 'pipeline', 'etl', 'analytics', 'report', 'dashboard'
        ]
        
        query_lower = query.lower()
        has_data_keywords = any(keyword in query_lower for keyword in data_keywords)
        
        logger.info(f"Fallback relevance classification - Is relevant: {has_data_keywords}")
        
        return RelevanceClassification(
            is_relevant=has_data_keywords,
            confidence=0.7 if has_data_keywords else 0.3,
            classification_reason="Fallback keyword detection",
            suggested_topics=["dataset information", "data sources", "table schemas"] if not has_data_keywords else []
        )

async def generate_greeting_response(query: str, base_url: str = None) -> str:
    """
    Generate a greeting response in the same language as the query using LLM
    """
    logger.info(f"Generating greeting response for query: '{query[:30]}{'...' if len(query) > 30 else ''}'")
    
    try:
        detected_language = await detect_language(query, base_url)
        logger.debug(f"Detected language for greeting: {detected_language}")
        
        llm = get_ollama_client(base_url)
        
        messages = [
            SystemMessage(content=f'''You are a helpful assistant for Suadeo Data Catalog. Generate a friendly greeting response in {detected_language}.
                    
                    The response should:
                    - Welcome the user to Suadeo Catalog Search
                    - Explain that you help with data catalog exploration and discovery
                    - List what kind of questions they can ask about:
                      • Dataset information and metadata
                      • Data sources and dataset_ids
                      • Content types and structures
                      • Catalog data like dataset status, description, owner info (name, email, account)
                      • Specific data they're looking for
                    - End with asking what they'd like to know about their data catalog
                    
                    Keep the tone friendly and professional. Respond entirely in {detected_language}.'''),
            HumanMessage(content=f'The user said: "{query}". Generate an appropriate greeting response.')
        ]
        
        logger.debug("Sending greeting response generation request to Ollama")
        response = llm.invoke(messages)
        greeting_response = response.content.strip()
        
        logger.info(f"Greeting response generated successfully in {detected_language}, length: {len(greeting_response)} characters")
        return greeting_response
        
    except Exception as e:
        logger.error(f"Error generating greeting response: {str(e)}")
        logger.warning("Falling back to default English greeting response")
        
        # Fallback to default English response
        return """Hello! Welcome to Suadeo Catalog Search. 
        I'm here to help you explore and discover data catalogs from Suadeo. You can ask me questions about:

        • Dataset information and metadata
        • Data sources and dataset_ids 
        • Content types and structures
        • Catalog data like dataset status, description, info about the owner like name, email, account

        • Specific data you're looking for

        Feel free to ask anything related to your Suadeo catalog - I'll search through the available data and provide you with relevant information and sources.

        What would you like to know about your data catalog today?"""

async def generate_irrelevant_response(query: str, suggested_topics: list[str] = None, base_url: str = settings.ollama_base_url) -> str:
    """
    Generate an irrelevant query response in the same language as the query using LLM
    """
    logger.info(f"Generating irrelevant response for query: '{query[:30]}{'...' if len(query) > 30 else ''}'")
    logger.debug(f"Suggested topics: {suggested_topics}")
    
    try:
        detected_language = await detect_language(query, base_url)
        logger.debug(f"Detected language for irrelevant response: {detected_language}")
        
        llm = get_ollama_client(base_url)
        
        topics_text = ""
        if suggested_topics:
            topics_text = f"Specifically, you could ask about: {', '.join(suggested_topics)}"
            logger.debug(f"Including suggested topics in response: {topics_text}")
        
        messages = [
            SystemMessage(content=f'''You are a helpful assistant for Suadeo Data Catalog. Generate a polite response in {detected_language} explaining that you only help with Suadeo data catalog queries.
                    
                    The response should:
                    - Politely explain that you specialize in Suadeo data catalog queries only
                    - List what kind of questions they can ask about:
                      • Dataset discovery and Suadeo Search
                      • Data source information and metadata
                      • Dataset information and metadata
                      • Data sources and dataset_ids
                      • Content types and structures
                      • Catalog data like dataset status, description, owner info (name, email, account)
                    - {topics_text if topics_text else "Provide examples like 'Show me Information about Dubai events' or 'What datasets are available for HR?'"}
                    - End with asking what they'd like to know about their data catalog
                    
                    Keep the tone helpful and redirect them positively. Respond entirely in {detected_language}.'''),
            HumanMessage(content=f'The user asked: "{query}". Generate an appropriate response explaining this is not relevant to data catalog queries.')
        ]
        
        logger.debug("Sending irrelevant response generation request to Ollama")
        response = llm.invoke(messages)
        irrelevant_response = response.content.strip()
        
        logger.info(f"Irrelevant response generated successfully in {detected_language}, length: {len(irrelevant_response)} characters")
        return irrelevant_response
        
    except Exception as e:
        logger.error(f"Error generating irrelevant response: {str(e)}")
        logger.warning("Falling back to default English irrelevant response")
        
        # Fallback to default English response
        base_message = """I'm specialized in helping you with Suadeo data catalog queries only. Please ask questions related to:

            • Dataset discovery and Suadeo Search
            • Data source information and metadata
            • Dataset information and metadata
            • Data sources and dataset_ids 
            • Content types and structures
            • Catalog data like dataset status, description, info about the owner like name, email, account etc

            Feel free to ask anything related to your Suadeo catalog - I'll search through the available data and provide you with relevant information and sources.
            What would you like to know about your data catalog today?"""

        if suggested_topics:
            suggestion_text = "\n\nFor example, you could ask about:\n" + "\n".join([f"• {topic}" for topic in suggested_topics])
            base_message += suggestion_text
            logger.debug("Added suggested topics to fallback response")
        else:
            base_message += "\n\nFor example: 'Show me Information about Dubai events' or 'What datasets are available for HR?'"

        return base_message

async def create_greeting_response(query: str, base_url: str =  settings.ollama_base_url) -> SuadeoSearchResponse:
    """
    Create a friendly greeting response for the Suadeo catalog using LLM
    """
    logger.info(f"Creating greeting response object for query: '{query[:30]}{'...' if len(query) > 30 else ''}'")
    
    welcome_message = await generate_greeting_response(query, base_url)
    
    response = SuadeoSearchResponse(
        answer=welcome_message,
        confidence=1.0,
        sources=[],
        has_sufficient_context=True,
        query="greeting",
        total_chunks_found=0
    )
    
    logger.info("Greeting response object created successfully")
    return response

async def create_irrelevant_response(query: str, suggested_topics: list[str] = None, base_url: str = settings.ollama_base_url) -> SuadeoSearchResponse:
    """
    Create a response for irrelevant queries using LLM
    """
    logger.info(f"Creating irrelevant response object for query: '{query[:30]}{'...' if len(query) > 30 else ''}'")
    
    response_message = await generate_irrelevant_response(query, suggested_topics, base_url)
    
    response = SuadeoSearchResponse(
        answer=response_message,
        confidence=1.0,
        sources=[],
        has_sufficient_context=True,
        query="irrelevant_query",
        total_chunks_found=0
    )
    
    logger.info("Irrelevant response object created successfully")
    return response