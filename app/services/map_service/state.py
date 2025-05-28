from typing import TypedDict, List, Dict, Any

class GeoState(TypedDict, total=False):
    query: str
    llm_output: Dict[str, Any]
    latitude: float
    longitude: float
    radius: int
    place_types: List[tuple]
    geojson: Dict[str, Any]
