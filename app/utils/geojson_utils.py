# import logging
# import folium
# import re
# import json
# import requests
# import time
# from typing import Dict, Any, List, Optional
# from math import radians, cos, sin, asin, sqrt

# from app.utils.logging import get_logger

# # Create a logger specific to this module
# logger = get_logger("app.services.geojson_utils")


# ## Version 1

# def extract_geojson_from_markdown(text: str) -> Dict[str, Any]:
#     pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
#     match = re.search(pattern, text, re.DOTALL)
#     if not match:
#         raise ValueError("No valid GeoJSON block found in the response.")
    
#     json_str = match.group(1).strip()
#     try:
#         return json.loads(json_str)
#     except json.JSONDecodeError as e:
#         raise ValueError(f"Invalid JSON format: {str(e)}")



# def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
#     """
#     Calculate the great circle distance between two points 
#     on the earth (specified in decimal degrees)
#     """
#     # Convert decimal degrees to radians
#     lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
#     # Haversine formula
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
#     c = 2 * asin(sqrt(a))
#     r = 6371  # Radius of earth in kilometers
#     return c * r

# def geocode_address(address: str) -> Optional[List[float]]:
#     """
#     Geocode an address using Nominatim API (OpenStreetMap)
#     Returns [longitude, latitude] if successful, None otherwise
#     """
#     # Adding user-agent is required by Nominatim's terms of use
#     headers = {
#         'User-Agent': 'GeoJSONValidator/1.0'
#     }
    
#     # Build the URL for the Nominatim API
#     base_url = "https://nominatim.openstreetmap.org/search"
#     params = {
#         'q': address,
#         'format': 'json',
#         'limit': 1
#     }
    
#     try:
#         response = requests.get(base_url, params=params, headers=headers)
#         response.raise_for_status()
#         results = response.json()
        
#         if results and len(results) > 0:
#             # Nominatim returns lat/lon, but GeoJSON uses lon/lat order
#             lat = float(results[0]['lat'])
#             lon = float(results[0]['lon'])
#             return [lon, lat]
#         else:
#             logger.warning(f"No geocoding results found for address: {address}")
#             return None
#     except Exception as e:
#         logger.error(f"Error during geocoding: {str(e)}")
#         return None

# def verify_and_correct_coordinates(geojson_data: Dict[str, Any], 
#                                   max_distance_km: float = 0.4,
#                                   delay_between_requests: float = 1.0) -> Dict[str, Any]:
#     """
#     Verify and correct coordinates in GeoJSON features using address information.
    
#     Args:
#         geojson_data: The GeoJSON data to verify
#         max_distance_km: Maximum allowed distance between provided and geocoded coordinates (in km)
#         delay_between_requests: Delay between geocoding requests to avoid rate limiting
        
#     Returns:
#         Updated GeoJSON with corrected coordinates where necessary
#     """
#     if geojson_data.get('type') != 'FeatureCollection':
#         logger.error("Input is not a valid GeoJSON FeatureCollection")
#         return geojson_data
    
#     # Create a copy of the input data to avoid modifying the original
#     corrected_geojson = geojson_data.copy()
    
#     corrections_made = 0
#     features_processed = 0
    
#     for i, feature in enumerate(corrected_geojson['features']):
#         features_processed += 1
        
#         # Skip features without properties or geometry
#         if not feature.get('properties') or not feature.get('geometry'):
#             continue
            
#         properties = feature['properties']
#         geometry = feature['geometry']
        
#         # Skip features without address information or not Point geometry
#         if geometry.get('type') != 'Point' or not properties.get('address'):
#             continue
            
#         address = properties['address']
#         original_coords = geometry['coordinates']
        
#         # Ensure proper coordinates format
#         if not isinstance(original_coords, list) or len(original_coords) < 2:
#             logger.warning(f"Invalid coordinates in feature {i}")
#             continue
            
#         logger.info(f"Verifying coordinates for address: {address}")
        
#         # Geocode the address
#         geocoded_coords = geocode_address(address)
        
#         # Add a delay to avoid hitting rate limits
#         time.sleep(delay_between_requests)
        
#         if not geocoded_coords:
#             logger.warning(f"Could not geocode address: {address}")
#             continue
            
#         # Calculate distance between original and geocoded coordinates
#         distance = haversine_distance(
#             original_coords[0], original_coords[1],
#             geocoded_coords[0], geocoded_coords[1]
#         )
        
#         # Log the comparison
#         logger.info(f"Original coordinates: {original_coords}")
#         logger.info(f"Geocoded coordinates: {geocoded_coords}")
#         logger.info(f"Distance between points: {distance:.2f} km")
        
#         # Update coordinates if they're too far from geocoded position
#         if distance > max_distance_km:
#             logger.warning(f"Coordinates mismatch detected (distance: {distance:.2f} km). Updating coordinates.")
#             geometry['coordinates'] = geocoded_coords
            
#             # Add metadata about the correction
#             if 'metadata' not in properties:
#                 properties['metadata'] = {}
#             properties['metadata']['coordinate_correction'] = {
#                 'original': original_coords,
#                 'corrected': geocoded_coords,
#                 'distance_km': round(distance, 3)
#             }
            
#             corrections_made += 1
    
#     logger.info(f"Verification complete. Processed {features_processed} features. Made {corrections_made} corrections.")
    
#     return corrected_geojson

# def visualize_geojson_comparison(original_geojson: Dict[str, Any], 
#                                 corrected_geojson: Dict[str, Any]) -> folium.Map:
#     """
#     Create a folium map to visualize the original and corrected coordinates
    
#     Args:
#         original_geojson: The original GeoJSON data
#         corrected_geojson: The corrected GeoJSON data
        
#     Returns:
#         A folium Map object
#     """
#     # Initialize map at the centroid of all points
#     all_lats = []
#     all_lons = []
    
#     # Collect all coordinates for centering the map
#     for feature in original_geojson.get('features', []):
#         if feature.get('geometry', {}).get('type') == 'Point':
#             coords = feature['geometry']['coordinates']
#             if len(coords) >= 2:
#                 all_lons.append(coords[0])
#                 all_lats.append(coords[1])
                
#     for feature in corrected_geojson.get('features', []):
#         if feature.get('geometry', {}).get('type') == 'Point':
#             coords = feature['geometry']['coordinates']
#             if len(coords) >= 2:
#                 all_lons.append(coords[0])
#                 all_lats.append(coords[1])
    
#     # Calculate center point
#     if all_lats and all_lons:
#         center_lat = sum(all_lats) / len(all_lats)
#         center_lon = sum(all_lons) / len(all_lons)
#     else:
#         # Default center (London)
#         center_lat = 51.5074
#         center_lon = -0.1278
        
#     # Create the map
#     m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
#     # Add original points in blue
#     for i, feature in enumerate(original_geojson.get('features', [])):
#         if feature.get('geometry', {}).get('type') == 'Point':
#             coords = feature['geometry']['coordinates']
#             if len(coords) >= 2:
#                 name = feature.get('properties', {}).get('name', f'Feature {i}')
#                 address = feature.get('properties', {}).get('address', 'No address')
#                 popup_text = f"<b>{name}</b><br>Address: {address}<br>Original Coordinates"
                
#                 folium.Marker(
#                     location=[coords[1], coords[0]],  # Folium uses [lat, lon]
#                     popup=popup_text,
#                     tooltip=f"{name} (Original)",
#                     icon=folium.Icon(color='blue', icon='info-sign')
#                 ).add_to(m)
    
#     # Add corrected points in red
#     for i, feature in enumerate(corrected_geojson.get('features', [])):
#         if feature.get('geometry', {}).get('type') == 'Point':
#             coords = feature['geometry']['coordinates']
#             metadata = feature.get('properties', {}).get('metadata', {})
            
#             # Only show corrected points that are different from original
#             if metadata.get('coordinate_correction'):
#                 name = feature.get('properties', {}).get('name', f'Feature {i}')
#                 address = feature.get('properties', {}).get('address', 'No address')
#                 distance = metadata['coordinate_correction'].get('distance_km', 'unknown')
                
#                 popup_text = f"<b>{name}</b><br>Address: {address}<br>Corrected Coordinates<br>Distance: {distance} km"
                
#                 folium.Marker(
#                     location=[coords[1], coords[0]],  # Folium uses [lat, lon]
#                     popup=popup_text,
#                     tooltip=f"{name} (Corrected)",
#                     icon=folium.Icon(color='red', icon='info-sign')
#                 ).add_to(m)
                
#                 # Draw a line between original and corrected points
#                 original_coords = metadata['coordinate_correction']['original']
#                 folium.PolyLine(
#                     locations=[
#                         [original_coords[1], original_coords[0]],
#                         [coords[1], coords[0]]
#                     ],
#                     color='purple',
#                     weight=2,
#                     opacity=0.7,
#                     tooltip=f"Correction for {name}"
#                 ).add_to(m)
    
#     return m


## Version 2
# import logging
# import folium
# import re
# import json
# import requests
# import time
# from typing import Dict, Any, List, Optional
# from math import radians, cos, sin, asin, sqrt

# from app.utils.logging import get_logger

# # Create a logger specific to this module
# logger = get_logger("app.services.geojson_utils")

# def extract_geojson_from_markdown(text: str) -> Dict[str, Any]:
#     pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
#     match = re.search(pattern, text, re.DOTALL)
#     if not match:
#         raise ValueError("No valid GeoJSON block found in the response.")
    
#     json_str = match.group(1).strip()
#     try:
#         return json.loads(json_str)
#     except json.JSONDecodeError as e:
#         raise ValueError(f"Invalid JSON format: {str(e)}")

# def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
#     """
#     Calculate the great circle distance between two points 
#     on the earth (specified in decimal degrees)
#     """
#     # Convert decimal degrees to radians
#     lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
#     # Haversine formula
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
#     c = 2 * asin(sqrt(a))
#     r = 6371  # Radius of earth in kilometers
#     return c * r

# def geocode_address(address: str) -> Optional[List[float]]:
#     """
#     Geocode an address using Nominatim API (OpenStreetMap)
#     Returns [longitude, latitude] if successful, None otherwise
#     """
#     # Adding user-agent is required by Nominatim's terms of use
#     headers = {
#         'User-Agent': 'GeoJSONValidator/1.0'
#     }
    
#     # Preprocess the address to make it Nominatim-friendly
#     # Remove shop numbers, specific tower details, and redundant terms
#     cleaned_address = re.sub(r'Shop\s*No\.\s*\d+\s*(&\s*\d+)?,?\s*', '', address)  # Remove "Shop No. X" or "Shop No. X & Y"
#     cleaned_address = re.sub(r',\s*Cluster\s*[A-Z],?', '', cleaned_address)  # Remove "Cluster X"
#     cleaned_address = re.sub(r',\s*Promenade\s*Level,?', '', cleaned_address)  # Remove "Promenade Level"
#     cleaned_address = re.sub(r',\s*The\s*O2\s*Residence,?', '', cleaned_address)  # Remove "The O2 Residence"
#     cleaned_address = re.sub(r',\s*Fortune\s*Executive\s*Tower,?', '', cleaned_address)  # Remove "Fortune Executive Tower"
#     cleaned_address = re.sub(r',\s*Saba\s*Tower\s*\d+,?', '', cleaned_address)  # Remove "Saba Tower X"
#     cleaned_address = re.sub(r',\s*One\s*Lake\s*Plaza,?', '', cleaned_address)  # Remove "One Lake Plaza"
#     cleaned_address = re.sub(r',\s*The\s*Park,?', '', cleaned_address)  # Remove "The Park"
#     cleaned_address = cleaned_address.strip(', ').rstrip(',')  # Clean up extra commas and spaces
    
#     # Ensure city and country are included
#     if 'Dubai' not in cleaned_address:
#         cleaned_address += ', Dubai'
#     if 'UAE' not in cleaned_address:
#         cleaned_address += ', UAE'
    
#     logger.info(f"Cleaned address for geocoding: {cleaned_address}")
    
#     # Build the URL for the Nominatim API
#     base_url = "https://nominatim.openstreetmap.org/search"
#     params = {
#         'q': cleaned_address,
#         'format': 'json',
#         'limit': 1,
#         'countrycodes': 'AE'  # Restrict to UAE
#     }
    
#     try:
#         response = requests.get(base_url, params=params, headers=headers)
#         response.raise_for_status()
#         results = response.json()
        
#         if results and len(results) > 0:
#             # Nominatim returns lat/lon, but GeoJSON uses lon/lat order
#             lat = float(results[0]['lat'])
#             lon = float(results[0]['lon'])
#             return [lon, lat]
#         else:
#             logger.warning(f"No geocoding results found for address: {cleaned_address}")
#             return None
#     except Exception as e:
#         logger.error(f"Error during geocoding: {str(e)}")
#         return None

# def verify_and_correct_coordinates(geojson_data: Dict[str, Any], 
#                                   max_distance_km: float = 0.4,
#                                   delay_between_requests: float = 1.0) -> Dict[str, Any]:
#     """
#     Verify and correct coordinates in GeoJSON features using address information.
    
#     Args:
#         geojson_data: The GeoJSON data to verify
#         max_distance_km: Maximum allowed distance between provided and geocoded coordinates (in km)
#         delay_between_requests: Delay between geocoding requests to avoid rate limiting
        
#     Returns:
#         Updated GeoJSON with coordinates corrected in-place where necessary
#     """
#     if geojson_data.get('type') != 'FeatureCollection':
#         logger.error("Input is not a valid GeoJSON FeatureCollection")
#         return geojson_data
    
#     # Work directly on the input data to modify coordinates in-place
#     corrections_made = 0
#     features_processed = 0
    
#     for i, feature in enumerate(geojson_data['features']):
#         features_processed += 1
        
#         # Skip features without properties or geometry
#         if not feature.get('properties') or not feature.get('geometry'):
#             continue
            
#         properties = feature['properties']
#         geometry = feature['geometry']
        
#         # Skip features without address information or not Point geometry
#         if geometry.get('type') != 'Point' or not properties.get('address'):
#             continue
            
#         address = properties['address']
#         original_coords = geometry['coordinates']
        
#         # Ensure proper coordinates format
#         if not isinstance(original_coords, list) or len(original_coords) < 2:
#             logger.warning(f"Invalid coordinates in feature {i}")
#             continue
            
#         logger.info(f"Verifying coordinates for address: {address}")
        
#         # Geocode the address
#         geocoded_coords = geocode_address(address)
        
#         # Add a delay to avoid hitting rate limits
#         time.sleep(delay_between_requests)
        
#         if not geocoded_coords:
#             logger.warning(f"Could not geocode address: {address}")
#             continue
            
#         # Calculate distance between original and geocoded coordinates
#         distance = haversine_distance(
#             original_coords[0], original_coords[1],
#             geocoded_coords[0], geocoded_coords[1]
#         )
        
#         # Log the comparison
#         logger.info(f"Original coordinates: {original_coords}")
#         logger.info(f"Geocoded coordinates: {geocoded_coords}")
#         logger.info(f"Distance between points: {distance:.2f} km")
        
#         # Update coordinates in-place if they're too far from geocoded position
#         if distance > max_distance_km:
#             logger.warning(f"Coordinates mismatch detected (distance: {distance:.2f} km). Updating coordinates.")
#             geometry['coordinates'] = geocoded_coords
#             corrections_made += 1
    
#     logger.info(f"Verification complete. Processed {features_processed} features. Made {corrections_made} corrections.")
    
#     return geojson_data

# def visualize_geojson_comparison(original_geojson: Dict[str, Any], 
#                                 corrected_geojson: Dict[str, Any]) -> folium.Map:
#     """
#     Create a folium map to visualize the original and corrected coordinates
    
#     Args:
#         original_geojson: The original GeoJSON data
#         corrected_geojson: The corrected GeoJSON data
        
#     Returns:
#         A folium Map object
#     """
#     # Initialize map at the centroid of all points
#     all_lats = []
#     all_lons = []
    
#     # Collect all coordinates for centering the map
#     for feature in original_geojson.get('features', []):
#         if feature.get('geometry', {}).get('type') == 'Point':
#             coords = feature['geometry']['coordinates']
#             if len(coords) >= 2:
#                 all_lons.append(coords[0])
#                 all_lats.append(coords[1])
                
#     for feature in corrected_geojson.get('features', []):
#         if feature.get('geometry', {}).get('type') == 'Point':
#             coords = feature['geometry']['coordinates']
#             if len(coords) >= 2:
#                 all_lons.append(coords[0])
#                 all_lats.append(coords[1])
    
#     # Calculate center point
#     if all_lats and all_lons:
#         center_lat = sum(all_lats) / len(all_lats)
#         center_lon = sum(all_lons) / len(all_lons)
#     else:
#         # Default center (London)
#         center_lat = 51.5074
#         center_lon = -0.1278
        
#     # Create the map
#     m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
#     # Add original points in blue
#     for i, feature in enumerate(original_geojson.get('features', [])):
#         if feature.get('geometry', {}).get('type') == 'Point':
#             coords = feature['geometry']['coordinates']
#             if len(coords) >= 2:
#                 name = feature.get('properties', {}).get('name', f'Feature {i}')
#                 address = feature.get('properties', {}).get('address', 'No address')
#                 popup_text = f"<b>{name}</b><br>Address: {address}<br>Original Coordinates"
                
#                 folium.Marker(
#                     location=[coords[1], coords[0]],  # Folium uses [lat, lon]
#                     popup=popup_text,
#                     tooltip=f"{name} (Original)",
#                     icon=folium.Icon(color='blue', icon='info-sign')
#                 ).add_to(m)
    
#     # Add corrected points in red
#     for i, feature in enumerate(corrected_geojson.get('features', [])):
#         if feature.get('geometry', {}).get('type') == 'Point':
#             coords = feature['geometry']['coordinates']
#             # Since metadata is no longer added, compare with original_geojson to detect corrections
#             original_feature = original_geojson.get('features', [])[i]
#             original_coords = original_feature.get('geometry', {}).get('coordinates', [])
#             if original_coords != coords:  # Check if coordinates were updated
#                 name = feature.get('properties', {}).get('name', f'Feature {i}')
#                 address = feature.get('properties', {}).get('address', 'No address')
#                 distance = haversine_distance(coords[0], coords[1], original_coords[0], original_coords[1]) if original_coords else 'unknown'
                
#                 popup_text = f"<b>{name}</b><br>Address: {address}<br>Corrected Coordinates<br>Distance: {distance:.2f} km"
                
#                 folium.Marker(
#                     location=[coords[1], coords[0]],  # Folium uses [lat, lon]
#                     popup=popup_text,
#                     tooltip=f"{name} (Corrected)",
#                     icon=folium.Icon(color='red', icon='info-sign')
#                 ).add_to(m)
                
#                 # Draw a line between original and corrected points
#                 folium.PolyLine(
#                     locations=[
#                         [original_coords[1], original_coords[0]],
#                         [coords[1], coords[0]]
#                     ],
#                     color='purple',
#                     weight=2,
#                     opacity=0.7,
#                     tooltip=f"Correction for {name}"
#                 ).add_to(m)
    
#     return m


## Version 3
#80233cd37c98454f9bc57964fe11baae

import logging
import folium
import re
import json
import time
from typing import Dict, Any, List, Optional
from math import radians, cos, sin, asin, sqrt
from opencage.geocoder import OpenCageGeocode
from pydantic import BaseModel
from ollama import chat
from langchain_ollama import ChatOllama
from app.utils.config import settings

from app.utils.logging import get_logger

# Create a logger specific to this module
logger = get_logger("app.services.geojson_utils")


class CountryCode(BaseModel):
    code: str  # ISO 3166-1 alpha-2 country code



# Initialize OpenCage Geocoder with your API key
OPENCAGE_API_KEY = "80233cd37c98454f9bc57964fe11baae"  # Replace with your OpenCage API key
geocoder = OpenCageGeocode(OPENCAGE_API_KEY)

def extract_geojson_from_markdown(text: str) -> Dict[str, Any]:
    pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError("No valid GeoJSON block found in the response.")
    
    json_str = match.group(1).strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")

def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def detect_country_code_with_ollama(address: str) -> str:
    """
    Use Ollama to detect the two-letter country code from a raw address string.
    """
    try:
        ## TODO: This chat will not work in the Linux VM as we are using ollama from a url, not localhost
        logger.info(f"Address for country detection: {address}")
        response = chat(
            model=settings.ollama_model,
            # base_url=settings.ollama_base_url,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Given the following address, return only the 2-letter ISO 3166-1 country code "
                        "corresponding to the country mentioned in the address. Do not include any explanation.\n\n"
                        f"Address: {address}"
                    ),
                }
            ],
            format=CountryCode.model_json_schema()
        )
        country = CountryCode.model_validate_json(response['message']['content'])
        # logger.info(f"Detected country code from LLM: {country.code}")
        return country.code.lower()
    except Exception as e:
        logger.warning(f"Ollama country detection failed: {e}")
        return "unknown"
    


def geocode_address(address: str) -> Optional[List[float]]:
    """
    Geocode an address using OpenCage Geocoding API.
    Returns [longitude, latitude] if successful, None otherwise.
    """
    cleaned_address = address.strip()
    cleaned_address = re.sub(r'\s+', ' ', cleaned_address)
    cleaned_address = cleaned_address.strip(', ').rstrip(',')

    logger.info(f"Cleaned address for geocoding: {cleaned_address}")

    # Detect country code using Ollama
    # country_code = detect_country_code_with_ollama(cleaned_address)
    # logger.info(f"Detected country code from LLM: {country_code}")

    try:
        # results = geocoder.geocode(cleaned_address, countrycode=country_code)
        results = geocoder.geocode(cleaned_address)
        if results and len(results) > 0:
            lat = float(results[0]['geometry']['lat'])
            lon = float(results[0]['geometry']['lng'])
            return [lon, lat]
        else:
            logger.warning(f"No geocoding results found for address: {cleaned_address}")
            return None
    except Exception as e:
        logger.error(f"Error during geocoding: {str(e)}")
        return None
    

def verify_and_correct_coordinates(geojson_data: Dict[str, Any], 
                                  max_distance_km: float = 0.4,
                                  delay_between_requests: float = 1.0) -> Dict[str, Any]:
    """
    Verify and correct coordinates in GeoJSON features using address information.
    
    Args:
        geojson_data: The GeoJSON data to verify
        max_distance_km: Maximum allowed distance between provided and geocoded coordinates (in km)
        delay_between_requests: Delay between geocoding requests to avoid rate limiting
        
    Returns:
        Updated GeoJSON with coordinates corrected in-place where necessary
    """
    if geojson_data.get('type') != 'FeatureCollection':
        logger.error("Input is not a valid GeoJSON FeatureCollection")
        return geojson_data
    
    # Work directly on the input data to modify coordinates in-place
    corrections_made = 0
    features_processed = 0
    
    for i, feature in enumerate(geojson_data['features']):
        features_processed += 1
        
        # Skip features without properties or geometry
        if not feature.get('properties') or not feature.get('geometry'):
            continue
            
        properties = feature['properties']
        geometry = feature['geometry']
        
        # Skip features without address information or not Point geometry
        if geometry.get('type') != 'Point' or not properties.get('address'):
            continue
            
        address = properties['address']
        original_coords = geometry['coordinates']
        
        # Ensure proper coordinates format
        if not isinstance(original_coords, list) or len(original_coords) < 2:
            logger.warning(f"Invalid coordinates in feature {i}")
            continue
            
        logger.info(f"Verifying coordinates for address: {address}")
        
        # Geocode the address
        geocoded_coords = geocode_address(address)
        
        # Add a delay to avoid hitting rate limits
        time.sleep(delay_between_requests)
        
        if not geocoded_coords:
            logger.warning(f"Could not geocode address: {address}")
            continue
            
        # Calculate distance between original and geocoded coordinates
        distance = haversine_distance(
            original_coords[0], original_coords[1],
            geocoded_coords[0], geocoded_coords[1]
        )
        
        # Log the comparison
        logger.info(f"Original coordinates: {original_coords}")
        logger.info(f"Geocoded coordinates: {geocoded_coords}")
        logger.info(f"Distance between points: {distance:.2f} km")
        
        # Update coordinates in-place if they're too far from geocoded position
        if distance > max_distance_km:
            logger.warning(f"Coordinates mismatch detected (distance: {distance:.2f} km). Updating coordinates.")
            geometry['coordinates'] = geocoded_coords
            corrections_made += 1
    
    logger.info(f"Verification complete. Processed {features_processed} features. Made {corrections_made} corrections.")
    
    return geojson_data

def visualize_geojson_comparison(original_geojson: Dict[str, Any], 
                                corrected_geojson: Dict[str, Any]) -> folium.Map:
    """
    Create a folium map to visualize the original and corrected coordinates
    
    Args:
        original_geojson: The original GeoJSON data
        corrected_geojson: The corrected GeoJSON data
        
    Returns:
        A folium Map object
    """
    # Initialize map at the centroid of all points
    all_lats = []
    all_lons = []
    
    # Collect all coordinates for centering the map
    for feature in original_geojson.get('features', []):
        if feature.get('geometry', {}).get('type') == 'Point':
            coords = feature['geometry']['coordinates']
            if len(coords) >= 2:
                all_lons.append(coords[0])
                all_lats.append(coords[1])
                
    for feature in corrected_geojson.get('features', []):
        if feature.get('geometry', {}).get('type') == 'Point':
            coords = feature['geometry']['coordinates']
            if len(coords) >= 2:
                all_lons.append(coords[0])
                all_lats.append(coords[1])
    
    # Calculate center point
    if all_lats and all_lons:
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
    else:
        # Default center (Dubai)
        center_lat = 25.0731
        center_lon = 55.1500
        
    # Create the map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
    
    # Track which features were corrected to avoid duplication
    corrected_features = set()
    
    # First, identify all corrected features
    for i, feature in enumerate(original_geojson.get('features', [])):
        if i < len(corrected_geojson.get('features', [])):
            corrected_feature = corrected_geojson.get('features', [])[i]
            
            # Check if this feature's coordinates were corrected
            if (feature.get('geometry', {}).get('type') == 'Point' and 
                corrected_feature.get('geometry', {}).get('type') == 'Point'):
                
                orig_coords = feature['geometry']['coordinates']
                corr_coords = corrected_feature['geometry']['coordinates']
                
                if orig_coords != corr_coords:
                    corrected_features.add(i)
    
    # Add original points in blue (only if not corrected)
    for i, feature in enumerate(original_geojson.get('features', [])):
        if feature.get('geometry', {}).get('type') == 'Point':
            coords = feature['geometry']['coordinates']
            if len(coords) >= 2:
                name = feature.get('properties', {}).get('name', f'Feature {i}')
                address = feature.get('properties', {}).get('address', 'No address')
                popup_text = f"<b>{name}</b><br>Address: {address}<br>Original Coordinates"
                
                # Always show original in blue
                folium.Marker(
                    location=[coords[1], coords[0]],  # Folium uses [lat, lon]
                    popup=popup_text,
                    tooltip=f"{name} (Original)",
                    icon=folium.Icon(color='blue', icon='info-sign')
                ).add_to(m)
    
    # Add corrected points in red (only for features that were actually corrected)
    for i in corrected_features:
        if i < len(corrected_geojson.get('features', [])):
            feature = corrected_geojson.get('features', [])[i]
            orig_feature = original_geojson.get('features', [])[i]
            
            if feature.get('geometry', {}).get('type') == 'Point':
                coords = feature['geometry']['coordinates']
                original_coords = orig_feature['geometry']['coordinates']
                
                name = feature.get('properties', {}).get('name', f'Feature {i}')
                address = feature.get('properties', {}).get('address', 'No address')
                distance = haversine_distance(coords[0], coords[1], original_coords[0], original_coords[1])
                
                popup_text = f"<b>{name}</b><br>Address: {address}<br>Corrected Coordinates<br>Distance: {distance:.2f} km"
                
                # Show corrected in red
                folium.Marker(
                    location=[coords[1], coords[0]],  # Folium uses [lat, lon]
                    popup=popup_text,
                    tooltip=f"{name} (Corrected)",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
                
                # Draw a line between original and corrected points
                folium.PolyLine(
                    locations=[
                        [original_coords[1], original_coords[0]],
                        [coords[1], coords[0]]
                    ],
                    color='purple',
                    weight=2,
                    opacity=0.7,
                    tooltip=f"Correction for {name}"
                ).add_to(m)
    
    return m