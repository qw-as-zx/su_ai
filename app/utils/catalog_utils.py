import requests
import json
import logging
from typing import Dict, Any, Optional

from app.utils.logging import get_logger

# Create a logger specific to this module
logger = get_logger("app.routers.suadeo_utils")

def get_token(base_url: str) -> Optional[str]:
    """
    Get authentication token from Suadeo API
    
    Args:
        base_url: Base URL for the API
        
    Returns:
        Authentication token or None if failed
    """
    payload = {
        "method": "POST",
        "endpoint": "api/Authentication/token",
        "data": "{\"userName\":\"demo\",\"password\":\"demo\",\"grantType\":\"password\",\"authenticationName\":\"DemoKeycloak\",\"licenseServerCode\":\"string\",\"clientId\":\"string\",\"accessToken\":\"string\"}",
        "dataType": "json",
        "contentType": "application/json"
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(base_url, json=payload, headers=headers)
        response.raise_for_status()
        
        logger.info("Token request successful!")
        token_response = response.json()
        
        # Extract token from response
        if isinstance(token_response, dict):
            for token_field in ['token', 'access_token', 'accessToken']:
                if token_field in token_response:
                    return token_response[token_field]
        
        logger.warning("Could not extract token from response")
        return None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting token: {e}")
        return None

def get_catalogs(base_url: str, token: str) -> Optional[Dict[str, Any]]:
    """
    Get catalogs using authentication token
    
    Args:
        base_url: Base URL for the API
        token: Authentication token
        
    Returns:
        Catalogs data or None if failed
    """
    payload = {
        "method": "GET",
        "endpoint": "api/userdata/new",
        "data": "{}",
        "dataType": "json",
        "contentType": "application/json"
    }
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    try:
        response = requests.post(base_url, json=payload, headers=headers)
        response.raise_for_status()
        
        logger.info("Catalogs request successful!")
        return response.json()
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting catalogs: {e}")
        return None

def fetch_catalog_data(api_url: str = "https://webclient-demo.suadeo.com/apiservice") -> Optional[Dict[str, Any]]:
    """
    Fetch complete catalog data from Suadeo API
    
    Args:
        api_url: Suadeo API URL
        
    Returns:
        Complete catalog data or None if failed
    """
    logger.info(f"Fetching catalog data from {api_url}")
    
    # Get token
    token = get_token(api_url)
    if not token:
        logger.error("Failed to get authentication token")
        return None
    
    # Get catalogs
    catalog_data = get_catalogs(api_url, token)
    if not catalog_data:
        logger.error("Failed to get catalog data")
        return None
    
    logger.info("Successfully fetched catalog data")
    return catalog_data