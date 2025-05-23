from fastapi import APIRouter, Depends, HTTPException, status
from app.models.schemas import WebSearchRequest, WebSearchResponse, MapSearchResponse
from app.services.web_search_services import WebSearchService

router = APIRouter(prefix="", tags=["Web Search"], responses={404: {"description": "Not found"}})



@router.post("/web_search", response_model=WebSearchResponse)
async def web_search(request: WebSearchRequest, web_search_service: WebSearchService = Depends()):
    try:
        result = await web_search_service.perform_web_search(
            query=request.query,
            search_context_size=request.search_context_size,
            user_id="default"  # Replace with actual user_id from auth if available
        )
        return WebSearchResponse(answer=result["text"], citations=result["citations"])
    except Exception as e:
        if "Rate limit exceeded" in str(e):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(e)
            )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Web search failed: {str(e)}")


## Fixed Strucutred with directly getting geojson V3
def get_web_search_service():
    return WebSearchService()

@router.post("/map_search", response_model=MapSearchResponse)
async def map_search(
    request: WebSearchRequest,
    web_search_service: WebSearchService = Depends(get_web_search_service)
):
    """
    Location-based search endpoint that returns GeoJSON-formatted data with citations.

    Example of Output
        ```json
        {
        "type": "FeatureCollection",
        "features": [
            {
            "type": "Feature",
            "properties": {
                "name": "Best Dental Clinic",
                "address": "123 ABCD Street, XYZ City"
            },
            "geometry": {
                "type": "Point",
                "coordinates": [38.7636, -9.0934]
            }
            }
        ]
        }```
    """
    try:
        result = await web_search_service.perform_map_search(
            query=request.query,
            user_id="default",
            # user_location=request.user_location
        )
        return MapSearchResponse(
            geojson=result["geojson"],
            citations=result["citations"]
            )
    except Exception as e:
        if "Rate limit exceeded" in str(e):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(e)
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Map search failed: {str(e)}"
        )