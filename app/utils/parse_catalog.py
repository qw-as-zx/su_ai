import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

from app.utils.logging import get_logger

# Create a logger specific to this module
logger = get_logger("app.routers.parse_catalog")

def parse_catalog_for_rag(catalog_response: List[Dict], source_endpoint: str,  source_type: str) -> Dict[str, Any]:
    """
    Parse catalog response into RAG-optimized format
    
    Args:
        catalog_response: Raw catalog response from API
        source_endpoint: API endpoint from which the data was fetched
        
    Returns:
        Dict containing structured data optimized for RAG applications
    """
    
    # Initialize the RAG-optimized structure
    rag_data = {
        "metadata": {
            "total_datasets": len(catalog_response),
            "parsed_at": datetime.now().isoformat(),
            "active_datasets": 0,
            "catalogs": set(),
            "domains": set(),
            "source_endpoint": source_endpoint,
            "source_endpoint_type":source_type
        },
        "datasets": [],
        "search_index": []
    }
    
    for item in catalog_response:
        # Skip items without proper structure
        if not isinstance(item, dict) or 'oid' not in item:
            continue
            
        # Count active datasets
        if item.get('status') == 2:
            rag_data["metadata"]["active_datasets"] += 1
            
        # Collect unique catalogs and domains
        if item.get('catalog'):
            rag_data["metadata"]["catalogs"].add(item['catalog'])
        
        domains = item.get('domains', [])
        for domain in domains:
            rag_data["metadata"]["domains"].add(domain)
        
        # Parse each dataset
        dataset = parse_single_dataset(item)
        rag_data["datasets"].append(dataset)
        
        # Create search index entries
        search_entries = create_search_entries(dataset, source_endpoint,source_type)
        rag_data["search_index"].extend(search_entries)
    
    # Convert sets to lists for JSON serialization
    rag_data["metadata"]["catalogs"] = list(rag_data["metadata"]["catalogs"])
    rag_data["metadata"]["domains"] = list(rag_data["metadata"]["domains"])
    
    return rag_data

def parse_single_dataset(item: Dict) -> Dict[str, Any]:
    """Parse a single dataset item"""
    
    # Extract multilingual descriptions
    descriptions = {}
    for desc in item.get('descriptions', []):
        lang_code = desc.get('isoCode', 'unknown')
        descriptions[lang_code] = desc.get('text', '')
    
    # Extract owner information
    owner = item.get('owner', {})
    owner_info = {
        "display_name": owner.get('displayName', ''),
        "account": owner.get('account', ''),
        "email": owner.get('eMail', ''),
        "authentication": owner.get('authentication', ''),
        "is_group": owner.get('isGroup', False)
    }
    
    # Extract stewards information
    stewards = []
    for steward in item.get('stewards', []):
        stewards.append({
            "display_name": steward.get('displayName', ''),
            "account": steward.get('account', ''),
            "email": steward.get('eMail', ''),
            "first_name": steward.get('fistName', ''),
            "last_name": steward.get('lastName', ''),
            "authentication": steward.get('authentication', '')
        })
    
    # Create the structured dataset
    dataset = {
        "id": item.get('oid'),
        "key": item.get('key', ''),
        "name": item.get('name', ''),
        "status": item.get('status', 0),
        "status_text": get_status_text(item.get('status', 0)),
        # "catalog": item.get('catalog'),
        'catalog': item.get('catalog', 'Unknown') or 'Unknown',
        "domains": item.get('domains', []),
        "descriptions": descriptions,
        "owner": owner_info,
        "stewards": stewards,
        "permissions": {
            "can_read": item.get('canRead', False),
            "can_edit": item.get('canEdit', False),
            "is_favorite": item.get('favorite', False)
        },
        "technical_info": {
            "type": item.get('type'),
            "rest_api": item.get('restAPI', False),
            "provider": item.get('provider'),
            "parent_key": item.get('parentKey'),
            "sort_index": item.get('sortIndex', 0)
        }
    }
    
    return dataset

def create_search_entries(dataset: Dict, source_endpoint: str, source_type: str) -> List[Dict]:
    """Create search index entries for RAG applications"""
    
    search_entries = []
    base_metadata = {
        "dataset_id": dataset["id"],
        "dataset_key": dataset["key"],
        "dataset_name": dataset["name"],
        "catalog": dataset["catalog"],
        "domains": dataset["domains"],
        "status": dataset["status_text"],
        "owner": dataset["owner"]["display_name"],
        "source_endpoint": source_endpoint,
        "source_endpoint_type": source_type
    }
    
    # Create entry for dataset name and basic info
    search_entries.append({
        "id": f"{dataset['id']}_basic",
        "content_type": "dataset_info",
        "content": f"Dataset: {dataset['name']}. Catalog: {dataset.get('catalog', 'N/A')}. Domains: {', '.join(dataset['domains'])}. Owner: {dataset['owner']['display_name']}",
        "metadata": base_metadata
    })
    
    # Create entries for each language description
    for lang_code, description in dataset["descriptions"].items():
        if description.strip():
            search_entries.append({
                "id": f"{dataset['id']}_{lang_code}",
                "content_type": "description",
                "language": lang_code,
                "content": description,
                "metadata": {**base_metadata, "language": lang_code}
            })
    
    # Create entry for stewards if any
    if dataset["stewards"]:
        steward_info = ", ".join([s["display_name"] for s in dataset["stewards"] if s["display_name"]])
        if steward_info:
            search_entries.append({
                "id": f"{dataset['id']}_stewards",
                "content_type": "stewards",
                "content": f"Data stewards for {dataset['name']}: {steward_info}",
                "metadata": base_metadata
            })
    
    return search_entries

def get_status_text(status_code: int) -> str:
    """Convert status code to readable text"""
    status_map = {
        0: "inactive",
        1: "pending",
        2: "active",
        3: "archived"
    }
    return status_map.get(status_code, "unknown")


def save_rag_data(rag_data: Dict, filename: str = "catalog_rag_data.json"):
    """
    Save the RAG-optimized data to a JSON file

    Args:
        rag_data: Dictionary containing RAG-optimized data
        filename: Path to the output JSON file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save the data to the JSON file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(rag_data, f, indent=2, ensure_ascii=False)
        logger.info(f"RAG data saved to {filename}")
        
    except Exception as e:
        logger.error(f"Failed to save RAG data to {filename}: {e}")
        raise IOError(f"Failed to save RAG data to {filename}: {str(e)}")
    

def get_search_content_only(rag_data: Dict) -> List[str]:
    """Extract just the content for vector embedding"""
    return [entry["content"] for entry in rag_data["search_index"]]

def filter_by_language(rag_data: Dict, language: str = "en") -> List[Dict]:
    """Filter search entries by language"""
    return [
        entry for entry in rag_data["search_index"]
        if entry.get("language") == language or entry.get("content_type") != "description"
    ]

def filter_by_catalog(rag_data: Dict, catalog_name: str) -> List[Dict]:
    """Filter datasets by catalog"""
    return [
        dataset for dataset in rag_data["datasets"]
        if dataset.get("catalog") == catalog_name
    ]

def get_active_datasets_only(rag_data: Dict) -> List[Dict]:
    """Get only active datasets (status = 2)"""
    return [
        dataset for dataset in rag_data["datasets"]
        if dataset.get("status") == 2
    ]