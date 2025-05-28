import overpy
from .state import GeoState

class OverpassHelper:
    @staticmethod
    def build_query(lat, lon, radius, kv_pairs):
        blocks = [
            f"""
            node["{key}"="{value}"](around:{radius},{lat},{lon});
            way["{key}"="{value}"](around:{radius},{lat},{lon});
            relation["{key}"="{value}"](around:{radius},{lat},{lon});
            """
            for key, value in kv_pairs
        ]
        return f"""
        (
          {''.join(blocks)}
        );
        out body;
        >;
        out skel qt;
        """

    @staticmethod
    def run_query(lat, lon, radius, kv_pairs):
        api = overpy.Overpass()
        query = OverpassHelper.build_query(lat, lon, radius, kv_pairs)
        features = []

        try:
            result = api.query(query)

            for node in result.nodes:
                name = node.tags.get("name", "").strip()
                if name:  # Filter out nodes without a name or empty name
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [float(node.lon), float(node.lat)]},
                        "properties": {"name": name, "tags": dict(node.tags)},
                    })

            for way in result.ways:
                name = way.tags.get("name", "").strip()
                if name and way.nodes:  # Filter out ways without a name or empty name
                    node = way.nodes[0]
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [float(node.lon), float(node.lat)]},
                        "properties": {"name": name, "tags": dict(way.tags)},
                    })

        except Exception as e:
            print(f"Overpass query error: {e}")

        return {"type": "FeatureCollection", "features": features}
