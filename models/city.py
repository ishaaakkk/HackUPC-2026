from pydantic import BaseModel, Field
from typing import Optional


class CityBase(BaseModel):
    geonames_id: int
    name: str
    country: str
    country_code: str
    lat: float
    lon: float
    population: int
    timezone: str


class WikipediaSummary(BaseModel):
    title: str
    extract: str                  # plain-text summary (~400 words)
    page_url: str
    thumbnail_url: Optional[str] = None


class OSMPoiCounts(BaseModel):
    # Nature / outdoor
    beaches: int = 0
    mountains: int = 0
    parks: int = 0
    forests: int = 0
    # Culture
    museums: int = 0
    galleries: int = 0
    theatres: int = 0
    historic_sites: int = 0
    # Nightlife / food
    bars: int = 0
    nightclubs: int = 0
    restaurants: int = 0
    cafes: int = 0
    # Accommodation
    hotels: int = 0
    hostels: int = 0
    # Practical
    airports: int = 0
    universities: int = 0


class CityRawData(BaseModel):
    """Everything collected before LLM enrichment."""
    city: CityBase
    wikipedia: Optional[WikipediaSummary] = None
    osm_pois: Optional[OSMPoiCounts] = None
    fetch_errors: list[str] = Field(default_factory=list)