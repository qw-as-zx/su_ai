from langgraph.graph import StateGraph, END
from .state import GeoState
from .llm_extractor import LLMExtractor
from .overpass import OverpassHelper

class GeoPipeline:
    def __init__(self):
        self.extractor = LLMExtractor()
        self.builder = StateGraph(GeoState)

        self.builder.add_node("LLM_Extraction", self.extractor.extract)
        self.builder.add_node("Parse_LLM", self.parse_location_data)
        self.builder.add_node("Query_Overpass", self.run_overpass_query)

        self.builder.set_entry_point("LLM_Extraction")
        self.builder.add_edge("LLM_Extraction", "Parse_LLM")
        self.builder.add_edge("Parse_LLM", "Query_Overpass")
        self.builder.add_edge("Query_Overpass", END)

        self.graph = self.builder.compile()

    def parse_location_data(self, state: GeoState) -> GeoState:
        data = state["llm_output"]
        place_types = [(d["key"].lower(), d["value"].lower()) for d in data["place_type"]]
        return {
            "latitude": data["latitude"],
            "longitude": data["longitude"],
            "radius": data.get("radius", 10000),
            "place_types": place_types,
            **state
        }

    def run_overpass_query(self, state: GeoState) -> GeoState:
        geojson = OverpassHelper.run_query(
            state["latitude"],
            state["longitude"],
            state["radius"],
            state["place_types"]
        )
        return {"geojson": geojson, **state}

    def run(self, query: str) -> GeoState:
        return self.graph.invoke({"query": query})
