"""
GeoNames data source.

GeoNames exposes a free REST API (requires a free account at geonames.org).
The `searchJSON` endpoint returns cities ranked by population so we can
trivially pull "top N cities in the world" or filter by country.

Docs: https://www.geonames.org/export/geonames-search.html
"""

import httpx
import asyncio
from typing import Optional
from models.city import CityBase


GEONAMES_BASE = "https://secure.geonames.org"

# Feature codes that mean "populated place"
# PPL  = populated place
# PPLA = seat of 1st-order admin division (state capital)
# PPLC = capital of a political entity
CITY_FEATURE_CODES = ["PPL", "PPLA", "PPLA2", "PPLC"]


class GeoNamesClient:
    """
    Async client for the GeoNames REST API.

    Usage:
        async with GeoNamesClient(username="your_username") as client:
            cities = await client.fetch_top_cities(limit=100)
    """

    def __init__(self, username: str, timeout: float = 10.0):
        self.username = username
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_top_cities(
        self,
        limit: int = 100,
        country_code: Optional[str] = None,   # e.g. "ES", "US"
        min_population: int = 100_000,
    ) -> list[CityBase]:
        """
        Return up to `limit` cities ranked by population descending.
        Pass country_code to scope to a single country.
        """
        # GeoNames max rows per request is 1000
        batch_size = min(limit, 1000)
        params = {
            "featureClass": "P",                    # P = populated places
            "orderby": "population",
            "maxRows": batch_size,
            "username": self.username,
            "type": "json",
        }
        if country_code:
            params["country"] = country_code
        if min_population:
            # GeoNames doesn't have a direct minPopulation param on searchJSON
            # but we can filter client-side; for very large pulls use the
            # cities1000.zip dump instead (see fetch_from_dump below).
            pass

        raw = await self._get("/searchJSON", params)
        geonames = raw.get("geonames", [])

        cities = []
        for g in geonames:
            pop = int(g.get("population") or 0)
            if pop < min_population:
                continue
            # Skip entries without a recognised city feature code
            if g.get("fcode") not in CITY_FEATURE_CODES:
                continue
            cities.append(self._parse(g))

        return cities[:limit]

    async def fetch_city_by_id(self, geonames_id: int) -> Optional[CityBase]:
        """Fetch a single city by its GeoNames ID."""
        raw = await self._get("/getJSON", {
            "geonameId": geonames_id,
            "username": self.username,
        })
        if "geonameId" not in raw:
            return None
        return self._parse(raw)

    # ------------------------------------------------------------------
    # Alternative: use the bulk dump (no API key rate limits)
    # ------------------------------------------------------------------

    @staticmethod
    def fetch_from_dump_url() -> str:
        """
        For large-scale ingestion (10 000+ cities) use the pre-built dumps:
          https://download.geonames.org/export/dump/cities500.zip   (~500 pop min)
          https://download.geonames.org/export/dump/cities1000.zip
          https://download.geonames.org/export/dump/cities5000.zip
          https://download.geonames.org/export/dump/cities15000.zip

        The tab-separated format is documented at:
          https://download.geonames.org/export/dump/readme.txt

        Columns (0-indexed):
          0  geonameid
          1  name
          4  latitude
          5  longitude
          8  country code
          14 timezone
          17 population
        """
        return "https://download.geonames.org/export/dump/cities5000.zip"

    @staticmethod
    def parse_dump_line(line: str) -> Optional[CityBase]:
        """Parse one TSV line from a GeoNames bulk dump file."""
        parts = line.strip().split("\t")
        if len(parts) < 18:
            return None
        try:
            return CityBase(
                geonames_id=int(parts[0]),
                name=parts[1],
                country=parts[8],       # ISO 3166-1 alpha-2 (e.g. "ES")
                country_code=parts[8],
                lat=float(parts[4]),
                lon=float(parts[5]),
                population=int(parts[14]) if parts[14] else 0,
                timezone=parts[17],
            )
        except (ValueError, IndexError):
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _get(self, path: str, params: dict) -> dict:
        assert self._client, "Use as async context manager"
        resp = await self._client.get(GEONAMES_BASE + path, params=params)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse(g: dict) -> CityBase:
        return CityBase(
            geonames_id=int(g.get("geonameId", 0)),
            name=g.get("name", ""),
            country=g.get("countryName", g.get("countryCode", "")),
            country_code=g.get("countryCode", ""),
            lat=float(g.get("lat", 0)),
            lon=float(g.get("lng", 0)),
            population=int(g.get("population") or 0),
            timezone=g.get("timezone", {}).get("timeZoneId", "") if isinstance(g.get("timezone"), dict) else g.get("timezone", ""),
        )


# ------------------------------------------------------------------
# Quick smoke test  (run: python data_sources/geonames.py)
# ------------------------------------------------------------------

async def _demo():
    """
    Demo using the public GeoNames demo account (rate-limited).
    Replace 'demo' with your free username from geonames.org for
    anything beyond quick testing.
    """
    async with GeoNamesClient(username="demo") as client:
        print("Fetching top 5 most populous cities via API...")
        cities = await client.fetch_top_cities(limit=5, min_population=1_000_000)
        for c in cities:
            print(f"  {c.name}, {c.country} — pop {c.population:,}  ({c.lat}, {c.lon})")

    print("\nDump URL for bulk ingestion:")
    print(" ", GeoNamesClient.fetch_from_dump_url())

    print("\nSample dump parse:")
    sample_line = "2643743\tLondon\t\t\t51.50853\t-0.12574\tP\tPPLC\tGB\t\tENG\t\t\t\t8961989\t\t\t0\tEurope/London\t2019-09-05"
    city = GeoNamesClient.parse_dump_line(sample_line)
    print(f"  Parsed: {city}")

if __name__ == "__main__":
    asyncio.run(_demo())