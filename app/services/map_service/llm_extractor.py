from langchain_core.output_parsers import JsonOutputParser
from .prompt import geo_prompt
from .llm_config import llm
from .state import GeoState

parser = JsonOutputParser()

class LLMExtractor:
    def __init__(self):
        self.chain = geo_prompt | llm | parser

    def extract(self, state: GeoState) -> GeoState:
        query = state["query"]
        structured = self.chain.invoke({"query": query})
        return {"llm_output": structured, **state}
