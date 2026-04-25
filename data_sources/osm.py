"""
OpenStreetMap data source via the Overpass API.

The Overpass API lets us count points of interest (POIs) within a radius
of any lat/lon. No API key required.

Public instances:
  https://overpass-api.de/api/interpreter   (primary)
  https://lz4.overpass-api.de/api/interpreter  (backup)

Rate limits: be polite — the service is free. We use:
  - A concurrency semaphore (default 2 simultaneous requests)
  - A delay between requests
  - The [timeout:25] Overpass directive to avoid blocking the server

Overpass QL cheat-sheet:
  node["amenity"="bar"](around:RADIUS,LAT,LON);  → nodes with that tag
  out count;  → return just the count, not the full geometry (cheap!)
"""

import httpx
import asyncio
from typing import Optional
from models.city import CityBase, OSMPoiCounts


OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]

# Search radius around the city centre, in metres.
# 15 km captures the metro area for most cities.
DEFAULT_RADIUS_M = 15_000

# OSM tag definitions for each POI category we care about.
# Each entry is (osm_key, osm_value).  We'll build one batched query.
POI_TAGS: dict[str, tuple[str, str]] = {
    # Nature
    "beaches":       ("natural",  "beach"),
    "mountains":     ("natural",  "peak"),
    "parks":         ("leisure",  "park"),
    "forests":       ("landuse",  "forest"),
    # Culture
    "museums":       ("tourism",  "museum"),
    "galleries":     ("tourism",  "gallery"),
    "theatres":      ("amenity",  "theatre"),
    "historic_sites":("historic", "yes"),       # any historic tag
    # Nightlife / food
    "bars":          ("amenity",  "bar"),
    "nightclubs":    ("amenity",  "nightclub"),
    "restaurants":   ("amenity",  "restaurant"),
    "cafes":         ("amenity",  "cafe"),
    # Accommodation
    "hotels":        ("tourism",  "hotel"),
    "hostels":       ("tourism",  "hostel"),
    # Practical
    "airports":      ("aeroway",  "aerodrome"),
    "universities":  ("amenity",  "university"),
}


def _build_overpass_query(lat: float, lon: float, radius_m: int) -> str:
    """
    Build a single batched Overpass QL query that counts all POI categories
    in one HTTP request using a union of `out count` sub-queries.

    Each union item is separated by a special comment so we can parse
    the counts back in order.
    """
    parts = []
    for field_name, (key, value) in POI_TAGS.items():
        tag_filter = f'["{key}"="{value}"]' if value != "yes" else f'["{key}"]'
        parts.append(
            f'/* {field_name} */\n'
            f'[out:json][timeout:25];\n'
            f'(\n'
            f'  node{tag_filter}(around:{radius_m},{lat},{lon});\n'
            f'  way{tag_filter}(around:{radius_m},{lat},{lon});\n'
            f');\n'
            f'out count;\n'
        )
    # We run them as separate queries rather than one union so each
    # count is unambiguous.  See fetch_poi_counts() for the batching logic.
    return parts


class OverpassClient:
    """
    Async client for the OpenStreetMap Overpass API.

    Fetches POI counts for each category defined in POI_TAGS.

    Usage:
        async with OverpassClient() as client:
            counts = await client.fetch_poi_counts(city)
    """

    def __init__(
        self,
        concurrency: int = 2,
        timeout: float = 30.0,
        radius_m: int = DEFAULT_RADIUS_M,
    ):
        self.semaphore = asyncio.Semaphore(concurrency)
        self.timeout = timeout
        self.radius_m = radius_m
        self._client: Optional[httpx.AsyncClient] = None
        self._endpoint_idx = 0     # round-robin between endpoints

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_poi_counts(
        self, city: CityBase, delay: float = 1.0
    ) -> OSMPoiCounts:
        """
        Fetch all POI counts for a city.
        Runs one Overpass request per POI category sequentially to be
        polite to the public instance. Takes ~16 s for a full city.

        For faster processing, run multiple cities concurrently via
        fetch_many(), not multiple categories per city.
        """
        async with self.semaphore:
            queries = _build_overpass_query(city.lat, city.lon, self.radius_m)
            counts: dict[str, int] = {}

            for field_name, query in zip(POI_TAGS.keys(), queries):
                count = await self._run_count_query(query)
                counts[field_name] = count
                await asyncio.sleep(delay)   # be polite to the public server

        return OSMPoiCounts(**counts)

    async def fetch_many(
        self,
        cities: list[CityBase],
        delay_between_cities: float = 2.0,
    ) -> dict[int, Optional[OSMPoiCounts]]:
        """
        Fetch POI counts for a list of cities.
        The concurrency semaphore in fetch_poi_counts() prevents overloading
        the Overpass server.
        """
        async def _fetch_one(city: CityBase):
            try:
                result = await self.fetch_poi_counts(city, delay=0.5)
                await asyncio.sleep(delay_between_cities)
                return city.geonames_id, result
            except Exception as e:
                print(f"  [OSM] Error fetching {city.name}: {e}")
                return city.geonames_id, None

        results = await asyncio.gather(*[_fetch_one(c) for c in cities])
        return dict(results)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run_count_query(self, query: str) -> int:
        """Execute one Overpass query and return the element count."""
        endpoint = OVERPASS_ENDPOINTS[self._endpoint_idx % len(OVERPASS_ENDPOINTS)]
        self._endpoint_idx += 1

        try:
            resp = await self._client.post(
                endpoint,
                data={"data": query},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
            data = resp.json()

            # Overpass `out count` response looks like:
            # {"elements": [{"type": "count", "tags": {"total": "42"}}]}
            elements = data.get("elements", [])
            if elements and elements[0].get("type") == "count":
                return int(elements[0].get("tags", {}).get("total", 0))
            return 0

        except (httpx.HTTPStatusError, httpx.RequestError, ValueError):
            return 0


# ------------------------------------------------------------------
# Quick smoke test  (run: python data_sources/osm.py)
# ------------------------------------------------------------------

async def _demo():
    from models.city import CityBase

    # Just test one city to avoid hammering the public Overpass instance
    barcelona = CityBase(
        geonames_id=3128760,
        name="Barcelona",
        country="Spain",
        country_code="ES",
        lat=41.3851,
        lon=2.1734,
        population=1_620_343,
        timezone="Europe/Madrid",
    )

    print(f"Fetching OSM POI counts for {barcelona.name} (radius={DEFAULT_RADIUS_M/1000:.0f} km)…")
    print("This will take ~30 s due to polite rate limiting.\n")

    async with OverpassClient(radius_m=DEFAULT_RADIUS_M) as client:
        counts = await client.fetch_poi_counts(barcelona, delay=0.5)

    print("POI counts:")
    for field, value in counts.model_dump().items():
        bar = "█" * min(value // 5, 40)
        print(f"  {field:<18} {value:>5}  {bar}")

if __name__ == "__main__":
    asyncio.run(_demo())