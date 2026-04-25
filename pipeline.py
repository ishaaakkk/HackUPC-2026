"""
Ingestion pipeline.

Orchestrates GeoNames → Wikipedia → OSM into CityRawData objects.
Designed for both small interactive runs and large batch jobs.

Usage examples:

  # Ingest top 50 world cities
  python pipeline.py --limit 50

  # Ingest from a pre-downloaded GeoNames dump file
  python pipeline.py --dump cities5000.txt --limit 200

  # Single city (useful for testing enrichment)
  python pipeline.py --city-id 2643743   # London
"""

import asyncio
import json
import sys
import argparse
from pathlib import Path
from typing import Optional

# Make sure the project root is on the path when running directly
sys.path.insert(0, str(Path(__file__).parent))

from models.city import CityBase, CityRawData
from data_sources.geonames import GeoNamesClient
from data_sources.wikipedia import WikipediaClient
from data_sources.osm import OverpassClient


OUTPUT_DIR = Path("raw_data")


class IngestionPipeline:
    """
    Pulls city data from all three sources and writes JSON files to disk.

    File layout:
      raw_data/
        london_2643743.json
        madrid_3117735.json
        ...

    Each file is a serialised CityRawData object ready for LLM enrichment.
    """

    def __init__(
        self,
        geonames_username: str = "demo",
        skip_osm: bool = False,          # OSM is slow; skip during fast tests
        osm_radius_m: int = 15_000,
        output_dir: Path = OUTPUT_DIR,
    ):
        self.geonames_username = geonames_username
        self.skip_osm = skip_osm
        self.osm_radius_m = osm_radius_m
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    async def ingest_cities(
        self,
        cities: list[CityBase],
        wiki_concurrency: int = 5,
    ) -> list[CityRawData]:
        """
        Enrich a list of CityBase objects with Wikipedia + OSM data.
        Returns CityRawData objects and writes them to disk.
        """
        print(f"\n{'─'*60}")
        print(f"  Ingesting {len(cities)} cities")
        print(f"  Wikipedia concurrency : {wiki_concurrency}")
        print(f"  OSM enabled           : {not self.skip_osm}")
        print(f"  Output dir            : {self.output_dir}")
        print(f"{'─'*60}\n")

        # ── Phase 1: Wikipedia (runs concurrently) ──────────────────────
        print("Phase 1/2  Fetching Wikipedia summaries…")
        async with WikipediaClient(concurrency=wiki_concurrency) as wiki:
            wiki_results = await wiki.fetch_many(cities)

        found = sum(1 for v in wiki_results.values() if v)
        print(f"  ✓ {found}/{len(cities)} summaries retrieved\n")

        # ── Phase 2: OSM (rate-limited, sequential per city) ─────────────
        osm_results: dict[int, Optional[object]] = {c.geonames_id: None for c in cities}
        if not self.skip_osm:
            print("Phase 2/2  Fetching OSM POI counts (this is slow — ~30 s/city)…")
            async with OverpassClient(radius_m=self.osm_radius_m) as osm:
                osm_results = await osm.fetch_many(cities, delay_between_cities=2.0)
            found_osm = sum(1 for v in osm_results.values() if v)
            print(f"  ✓ {found_osm}/{len(cities)} OSM POI sets retrieved\n")
        else:
            print("Phase 2/2  OSM skipped.\n")

        # ── Assemble CityRawData ─────────────────────────────────────────
        results = []
        for city in cities:
            raw = CityRawData(
                city=city,
                wikipedia=wiki_results.get(city.geonames_id),
                osm_pois=osm_results.get(city.geonames_id),
            )
            if not raw.wikipedia:
                raw.fetch_errors.append("wikipedia: not found")
            if not self.skip_osm and not raw.osm_pois:
                raw.fetch_errors.append("osm: fetch failed")

            self._save(raw)
            results.append(raw)

        self._print_summary(results)
        return results

    async def ingest_from_api(
        self,
        limit: int = 50,
        country_code: Optional[str] = None,
        min_population: int = 500_000,
        **kwargs,
    ) -> list[CityRawData]:
        """Convenience: fetch city list from GeoNames API then ingest."""
        print(f"Fetching city list from GeoNames API (limit={limit})…")
        async with GeoNamesClient(username=self.geonames_username) as geo:
            cities = await geo.fetch_top_cities(
                limit=limit,
                country_code=country_code,
                min_population=min_population,
            )
        print(f"  ✓ {len(cities)} cities fetched from GeoNames\n")
        return await self.ingest_cities(cities, **kwargs)

    async def ingest_from_dump(
        self,
        dump_path: Path,
        limit: int = 100,
        min_population: int = 500_000,
        **kwargs,
    ) -> list[CityRawData]:
        """Parse a local GeoNames dump file and ingest the top cities."""
        cities = []
        print(f"Reading GeoNames dump: {dump_path}")
        with open(dump_path, encoding="utf-8") as f:
            for line in f:
                city = GeoNamesClient.parse_dump_line(line)
                if city and city.population >= min_population:
                    cities.append(city)

        # Sort by population descending, take top N
        cities.sort(key=lambda c: c.population, reverse=True)
        cities = cities[:limit]
        print(f"  ✓ {len(cities)} cities loaded (pop ≥ {min_population:,})\n")
        return await self.ingest_cities(cities, **kwargs)

    # ------------------------------------------------------------------

    def _save(self, raw: CityRawData):
        slug = f"{raw.city.name.lower().replace(' ', '_')}_{raw.city.geonames_id}"
        path = self.output_dir / f"{slug}.json"
        path.write_text(raw.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def _print_summary(results: list[CityRawData]):
        total       = len(results)
        with_wiki   = sum(1 for r in results if r.wikipedia)
        with_osm    = sum(1 for r in results if r.osm_pois)
        with_errors = sum(1 for r in results if r.fetch_errors)

        print("─" * 60)
        print(f"  Ingestion complete")
        print(f"  Total cities   : {total}")
        print(f"  With Wikipedia : {with_wiki}/{total}")
        print(f"  With OSM POIs  : {with_osm}/{total}")
        print(f"  With errors    : {with_errors}/{total}")
        if with_errors:
            for r in results:
                if r.fetch_errors:
                    print(f"    {r.city.name}: {', '.join(r.fetch_errors)}")
        print("─" * 60)


# ------------------------------------------------------------------
# Demo run (no GeoNames API key needed — uses hardcoded sample cities)
# ------------------------------------------------------------------

SAMPLE_CITIES = [
    CityBase(geonames_id=2643743, name="London",        country="United Kingdom", country_code="GB", lat=51.5074, lon=-0.1278,  population=8_961_989, timezone="Europe/London"),
    CityBase(geonames_id=3117735, name="Madrid",        country="Spain",          country_code="ES", lat=40.4168, lon=-3.7038,  population=3_223_334, timezone="Europe/Madrid"),
    CityBase(geonames_id=1850147, name="Tokyo",         country="Japan",          country_code="JP", lat=35.6762, lon=139.6503, population=13_960_000,timezone="Asia/Tokyo"),
    CityBase(geonames_id=2988507, name="Paris",         country="France",         country_code="FR", lat=48.8566, lon=2.3522,   population=2_148_271, timezone="Europe/Paris"),
    CityBase(geonames_id=3128760, name="Barcelona",     country="Spain",          country_code="ES", lat=41.3851, lon=2.1734,   population=1_620_343, timezone="Europe/Madrid"),
]


async def _demo():
    pipeline = IngestionPipeline(
        geonames_username="demo",
        skip_osm=True,    # skip OSM to keep demo fast
        output_dir=Path("raw_data"),
    )
    results = await pipeline.ingest_cities(SAMPLE_CITIES, wiki_concurrency=3)

    # Print a sample of what was collected
    print("\nSample output (London):")
    london = next(r for r in results if r.city.name == "London")
    if london.wikipedia:
        print(f"  Wikipedia extract ({len(london.wikipedia.extract.split())} words):")
        print(f"  {london.wikipedia.extract[:200]}…")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

async def _main():
    parser = argparse.ArgumentParser(description="Travel platform city ingestion")
    parser.add_argument("--limit",    type=int, default=10)
    parser.add_argument("--dump",     type=str, help="Path to GeoNames dump file")
    parser.add_argument("--city-id",  type=int, help="Ingest a single GeoNames city ID")
    parser.add_argument("--country",  type=str, help="ISO country code filter (e.g. ES)")
    parser.add_argument("--skip-osm", action="store_true")
    parser.add_argument("--geonames-user", default="demo")
    args = parser.parse_args()

    pipeline = IngestionPipeline(
        geonames_username=args.geonames_user,
        skip_osm=args.skip_osm,
    )

    if args.city_id:
        async with GeoNamesClient(username=args.geonames_user) as geo:
            city = await geo.fetch_city_by_id(args.city_id)
        if not city:
            print(f"City ID {args.city_id} not found in GeoNames.")
            return
        await pipeline.ingest_cities([city])
    elif args.dump:
        await pipeline.ingest_from_dump(Path(args.dump), limit=args.limit)
    else:
        await pipeline.ingest_from_api(
            limit=args.limit,
            country_code=args.country,
        )


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        asyncio.run(_demo())
    else:
        asyncio.run(_main())