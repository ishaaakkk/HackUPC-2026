from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from app.stt import transcribe_audio
from app.llm import refine_locations_with_voice
import json
import requests as http_requests
import tempfile
import urllib.request
import os
import re
from pathlib import Path
from typing import Any, Dict, List

router = APIRouter(prefix="/api")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------- helpers ----------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def is_demo_mode() -> bool:
    return os.getenv("DEMO_MODE", "0").lower() in ("1", "true", "yes")


def _parse_destinations(destinations: str) -> List[Any]:
    """Accepts either a comma-separated string or a JSON array of destination objects."""
    raw = (destinations or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return [d.strip() for d in raw.split(",") if d.strip()]


_POSTAL_CODE_RE = re.compile(r"\b\d{4,6}\b")


def _clean_destination_piece(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,")


def _norm_key(value: Any) -> str:
    """Stable key for comparing cities/countries from user/API strings."""
    import unicodedata

    text = _clean_destination_piece(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _same_place_name(a: Any, b: Any) -> bool:
    """True when two city/airport query strings clearly mean the same place."""
    ak = _norm_key(a)
    bk = _norm_key(b)
    if not ak or not bk:
        return False
    return ak == bk or ak in bk or bk in ak


def _country_from_formatted_address(address: str) -> str:
    parts = [_clean_destination_piece(p) for p in str(address or "").split(",") if _clean_destination_piece(p)]
    return parts[-1] if parts else ""


def _city_from_formatted_address(address: str) -> str:
    """Best-effort city extraction from a Google formatted_address string.

    Google Places often returns addresses like:
    "C/ de Mallorca, 401, L'Eixample, 08013 Barcelona, Spain".
    For flights, Skyscanner normally needs "Barcelona", not the landmark name.
    """
    parts = [_clean_destination_piece(p) for p in str(address or "").split(",") if _clean_destination_piece(p)]
    if len(parts) < 2:
        return ""

    for part in reversed(parts[:-1]):
        cleaned = _POSTAL_CODE_RE.sub("", part)
        cleaned = _clean_destination_piece(cleaned)
        lower = cleaned.lower()
        if not cleaned:
            continue
        if any(street_word in lower for street_word in ["street", "st ", "avenue", "road", "c/", "carrer", "calle", "via "]):
            continue
        return cleaned
    return ""


def _destination_to_flight_query(destination: Any) -> str:
    """Convert a detected place object into something useful for flight search."""
    if isinstance(destination, str):
        return _clean_destination_piece(destination)
    if not isinstance(destination, dict):
        return _clean_destination_piece(destination)

    explicit = _clean_destination_piece(destination.get("flight_search_city"))
    if explicit:
        return explicit

    address = _clean_destination_piece(destination.get("formatted_address"))
    city_from_address = _city_from_formatted_address(address)
    if city_from_address:
        return city_from_address

    for key in ("destination_name", "city", "name", "country"):
        value = _clean_destination_piece(destination.get(key))
        if value:
            return value
    return ""


def _destination_country(destination: Any) -> str:
    """Best-effort country extraction used to avoid repeated flight cards by country."""
    if isinstance(destination, dict):
        for key in ("flight_search_country", "country"):
            value = _clean_destination_piece(destination.get(key))
            if value:
                return value
        address_country = _country_from_formatted_address(str(destination.get("formatted_address") or ""))
        if address_country:
            return address_country
    return ""


def _destination_display_name(destination: Any) -> str:
    if isinstance(destination, dict):
        for key in ("name", "city", "destination_name", "flight_search_city"):
            value = _clean_destination_piece(destination.get(key))
            if value:
                return value
    return _clean_destination_piece(destination)


def _normalize_destinations_for_flights(destinations: List[Any], origin: str = "") -> Dict[str, Any]:
    """Build the flight search list and metadata.

    - Same city as origin is not sent to Skyscanner; it becomes a no-flight-needed result.
    - Duplicate cities are removed.
    - Duplicate countries are removed by default. Set DEDUPE_FLIGHT_COUNTRIES=0 to disable it.
    """
    flight_destinations: List[str] = []
    no_flight_needed: List[Dict[str, Any]] = []
    skipped_duplicates: List[Dict[str, Any]] = []
    mappings: List[Dict[str, Any]] = []

    seen_cities = set()
    seen_countries = set()
    dedupe_countries = os.getenv("DEDUPE_FLIGHT_COUNTRIES", "1").strip().lower() not in {"0", "false", "no"}

    for destination in destinations:
        flight_query = _destination_to_flight_query(destination)
        if not flight_query:
            continue

        display_name = _destination_display_name(destination)
        country = _destination_country(destination)
        city_key = _norm_key(flight_query)
        country_key = _norm_key(country)

        record = {
            "display_name": display_name,
            "flight_search_city": flight_query,
            "country": country,
            "original": destination,
        }

        if _same_place_name(origin, flight_query):
            no_flight_needed.append({
                **record,
                "status": "no_flight_needed",
                "message": "No flight needed",
                "reason": "Destination is the same as origin.",
            })
            mappings.append({**record, "action": "no_flight_needed_same_as_origin"})
            if dedupe_countries and country_key:
                seen_countries.add(country_key)
            continue

        if city_key in seen_cities:
            skipped_duplicates.append({**record, "reason": "duplicate_city"})
            mappings.append({**record, "action": "skipped_duplicate_city"})
            continue

        if dedupe_countries and country_key and country_key in seen_countries:
            skipped_duplicates.append({**record, "reason": "duplicate_country"})
            mappings.append({**record, "action": "skipped_duplicate_country"})
            continue

        seen_cities.add(city_key)
        if dedupe_countries and country_key:
            seen_countries.add(country_key)
        flight_destinations.append(flight_query)
        mappings.append({**record, "action": "flight_search"})

    return {
        "flight_destinations": flight_destinations,
        "no_flight_needed": no_flight_needed,
        "skipped_duplicates": skipped_duplicates,
        "mappings": mappings,
        "dedupe_countries_enabled": dedupe_countries,
    }


# ---------- PHASE 1: Media Analysis ----------

@router.post("/analyze-media")
async def analyze_media(
    media: UploadFile = File(None),
    url: str = Form(None)
):
    """Receives an image/video file or URL and analyzes it with Vision + Places."""
    if is_demo_mode():
        mock_path = PROJECT_ROOT / "output_location.json"
        if mock_path.exists():
            with open(mock_path, "r") as f:
                return json.load(f)
        else:
            return {"note": "DEMO_MODE active but output_location.json not found"}

    if not media and not url:
        raise HTTPException(status_code=400, detail="Must provide either media file or url")

    tmp_file = None
    try:
        suffix = ".jpg"
        mime_type = "image/jpeg"

        if media and media.filename:
            original_name = media.filename
            lower = original_name.lower()
            if lower.endswith(".png"):
                suffix = ".png"
            elif lower.endswith(".webp"):
                suffix = ".webp"
            elif lower.endswith((".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v")):
                suffix = Path(lower).suffix or ".mp4"
            if media.content_type:
                mime_type = media.content_type
        elif url:
            lower = url.lower().split("?")[0]
            if lower.endswith(".mp4"):
                suffix = ".mp4"; mime_type = "video/mp4"
            elif lower.endswith(".mov"):
                suffix = ".mov"; mime_type = "video/quicktime"
            elif lower.endswith(".webm"):
                suffix = ".webm"; mime_type = "video/webm"
            elif lower.endswith(".png"):
                suffix = ".png"; mime_type = "image/png"
            elif lower.endswith(".webp"):
                suffix = ".webp"; mime_type = "image/webp"

        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        tmp_file = tmp_path

        source_input = "uploaded_media"
        source_type = "file_upload"

        if media:
            content = await media.read()
            with open(tmp_path, "wb") as f:
                f.write(content)
            source_input = media.filename or "uploaded_media"
            source_type = "file_upload"
        elif url:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as response:
                content = response.read()
                content_type = response.headers.get("Content-Type")
                if content_type and not mime_type.startswith("video/"):
                    mime_type = content_type.split(";")[0].strip() or mime_type
            with open(tmp_path, "wb") as f:
                f.write(content)
            source_input = url
            source_type = "url"

        from app.vision_places import analyze_media_with_vision_places
        result = analyze_media_with_vision_places(
            tmp_path,
            mime_type=mime_type,
            source_input=source_input,
            source_type=source_type,
            max_candidates=_env_int("MAX_CANDIDATES", 5),
            photos_per_place=_env_int("PHOTOS_PER_PLACE", 1),
            output_dir=PROJECT_ROOT,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing media: {str(e)}")
    finally:
        if tmp_file and os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass


@router.get("/detect-origin")
async def detect_origin(request: Request):
    """Detects the user's origin city based on their IP address."""
    try:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.headers.get("x-real-ip") or request.client.host

        if client_ip and client_ip.startswith("::ffff:"):
            client_ip = client_ip[7:]

        local_ips = {"::1", "127.0.0.1", "localhost"}
        is_local = (
            client_ip in local_ips
            or (client_ip and client_ip.startswith(("10.", "192.168.", "172.")))
        )

        if is_local:
            return {
                "status": "local",
                "city": "Barcelona",
                "country": "Spain",
                "country_code": "ES",
                "latitude": 41.3874,
                "longitude": 2.1686,
                "note": "Local IP detected, using Barcelona by default",
            }

        resp = http_requests.get(
            f"https://ipapi.co/{client_ip}/json/",
            headers={"User-Agent": "HackUPC-Travel-App/1.0"},
            timeout=5,
        )
        data = resp.json()

        if resp.status_code != 200 or data.get("error"):
            return {
                "status": "error",
                "city": "Barcelona",
                "country": "Spain",
                "country_code": "ES",
                "latitude": 41.3874,
                "longitude": 2.1686,
                "note": "Could not detect location, using Barcelona by default",
            }

        return {
            "status": "ok",
            "city": data.get("city", "Unknown"),
            "country": data.get("country_name", "Unknown"),
            "country_code": data.get("country_code", ""),
            "latitude": data.get("latitude", 0),
            "longitude": data.get("longitude", 0),
        }
    except Exception as e:
        return {
            "status": "error",
            "city": "Barcelona",
            "country": "Spain",
            "country_code": "ES",
            "latitude": 41.3874,
            "longitude": 2.1686,
            "note": f"Error: {str(e)}",
        }


# ---------- PHASE 2: Voice Validation ----------

@router.post("/voice-validate")
async def voice_validate(
    audio: UploadFile = File(None),
    transcript: str = Form(None),
    locations: str = Form(...)
):
    try:
        locations_list = json.loads(locations)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="'locations' is not valid JSON")

    try:
        if not transcript and audio:
            # Fallback to backend transcription if no transcript provided from frontend
            transcript = transcribe_audio(audio.file)
        
        if not transcript:
            raise HTTPException(status_code=400, detail="No transcript or audio provided")

        result = refine_locations_with_voice(locations_list, transcript)
        refined = result.get("locations", locations_list)
        # Preserve flight hints when Gemini returns only old fields.
        by_name = {str(x.get("city") or x.get("name", "")).lower(): x for x in locations_list if isinstance(x, dict)}
        for loc in refined:
            if not isinstance(loc, dict):
                continue
            old = by_name.get(str(loc.get("city") or loc.get("name", "")).lower())
            if old:
                loc.setdefault("flight_search_city", old.get("flight_search_city"))
                loc.setdefault("formatted_address", old.get("formatted_address"))
                loc.setdefault("maps_url", old.get("maps_url"))
        return {"transcript": transcript, "locations": refined}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing voice: {str(e)}")


# ---------- PHASE 3: Flight Search ----------

@router.get("/search-flights")
def search_flights(origin: str, destinations: str, date: str = "2026"):
    """Searches flights. Destinations can be names or JSON objects from the image-analysis step."""
    from flights import SkyscannerOptimizer
    from hotels import HotelSearcher

    if is_demo_mode():
        mock_path = Path(__file__).parent / "mock_flights.json"
        if mock_path.exists():
            with open(mock_path, "r") as f:
                return json.load(f)
        else:
            raise HTTPException(status_code=500, detail="DEMO_MODE active but mock_flights.json not found")

    api_key = os.getenv("SKYSCANNER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="SKYSCANNER_API_KEY not configured")

    original_dest_list = _parse_destinations(destinations)
    if not original_dest_list:
        raise HTTPException(status_code=400, detail="No destinations provided")

    normalization = _normalize_destinations_for_flights(original_dest_list, origin=origin)
    flight_dest_list = normalization["flight_destinations"]

    optimizer = SkyscannerOptimizer(api_key)
    hotel_searcher = HotelSearcher(api_key)

    try:
        if flight_dest_list:
            results = optimizer.optimize_route([origin] + flight_dest_list, date)
        else:
            # All detected destinations are the origin itself or were removed as
            # duplicates. Return a normal payload instead of forcing a "no flights" UI.
            results = {"origin": origin, "results": {}, "route": [origin]}

        results["flight_destinations_used"] = flight_dest_list
        results["original_detected_destinations"] = original_dest_list
        results["destination_mappings"] = normalization["mappings"]
        results["skipped_duplicate_destinations"] = normalization["skipped_duplicates"]
        results["dedupe_flight_countries_enabled"] = normalization["dedupe_countries_enabled"]
        results["destination_normalization_note"] = (
            "Detected image places are converted to city/airport-friendly names before flight search. "
            "Destinations equal to the origin are marked as 'No flight needed'. "
            "Duplicate flight cities and duplicate countries are removed before calling Skyscanner."
        )

        result_items = results.setdefault("results", {})

        for item in normalization["no_flight_needed"]:
            key = item.get("display_name") or item.get("flight_search_city") or origin
            result_items[key] = {
                "destination_name": key,
                "flight_search_city": item.get("flight_search_city"),
                "country": item.get("country"),
                "status": "no_flight_needed",
                "message": "No flight needed",
                "reason": item.get("reason"),
                "requires_flight": False,
                "price": None,
                "flight_price": None,
                "flights": [],
            }

        for dest_name, info in result_items.items():
            if not isinstance(info, dict):
                continue
            if info.get("status") == "no_flight_needed" or info.get("requires_flight") is False:
                # Avoid the frontend interpreting this as "sin vuelos en los próximos meses".
                info.setdefault("message", "No flight needed")
                info.setdefault("flights", [])
                continue
            hotel_key = info.get("flight_search_city") or info.get("destination_name") or dest_name
            info["hotel_price"] = hotel_searcher.get_hotel_prices(hotel_key)
        return results
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching flights: {str(e)}")

# ---------------------------------------------------------------------------
# Final destination-country cleanup
# ---------------------------------------------------------------------------
# The analyzer should already return valid countries, but keep /search-flights
# defensive so postal codes or state abbreviations are never treated as countries.

_BAD_COUNTRY_VALUES_FINAL = {
    "", "unknown", "n/a", "na", "none", "null", "01001",
    "ca", "wa", "california", "washington", "mt everest", "mount everest", "everest",
}

_COUNTRY_ALIASES_FINAL = {
    "united states": "USA",
    "united states of america": "USA",
    "us": "USA",
    "u s": "USA",
    "u s a": "USA",
    "usa": "USA",
    "uk": "United Kingdom",
    "u k": "United Kingdom",
}


def _clean_country_final(value: Any) -> str:
    raw = _clean_destination_piece(value)
    key = _norm_key(raw)
    if not key or key in _BAD_COUNTRY_VALUES_FINAL or re.fullmatch(r"\d{3,10}", key.replace(" ", "")):
        return ""
    return _COUNTRY_ALIASES_FINAL.get(key, raw)


def _destination_country(destination: Any) -> str:
    """Final override: robust country extraction for flight dedupe."""
    if isinstance(destination, dict):
        # Prefer the formatted address / known country over bad old fields.
        address_country = _clean_country_final(_country_from_formatted_address(str(destination.get("formatted_address") or "")))
        if address_country:
            return address_country
        for key in ("flight_search_country", "country"):
            value = _clean_country_final(destination.get(key))
            if value:
                return value
    return ""


# ---------------------------------------------------------------------------
# V14 backend flight normalization override
# ---------------------------------------------------------------------------
# Keep bad country fragments/postal codes out of flight search metadata and use
# airport-friendly destination cities for known POIs such as Mount Everest.

try:
    import pycountry as _pycountry_main_v14  # type: ignore
except Exception:  # pragma: no cover
    _pycountry_main_v14 = None

_COUNTRY_ALIASES_MAIN_V14 = {
    "usa": "United States",
    "u.s.a.": "United States",
    "u.s.": "United States",
    "us": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "uae": "United Arab Emirates",
    "mt everest": "",
    "mount everest": "",
    "everest": "",
    "ca": "",
    "wa": "",
    "01001": "",
}

_BAD_CITY_MAIN_V14 = {
    "ca", "wa", "california", "washington", "mt everest", "mount everest", "everest", "01001",
}

_PLACE_HINTS_MAIN_V14 = [
    ("mount everest", "Nepal", "Kathmandu"),
    ("mt everest", "Nepal", "Kathmandu"),
    ("everest", "Nepal", "Kathmandu"),
    ("sagarmatha", "Nepal", "Kathmandu"),
    ("mount fuji", "Japan", "Tokyo"),
    ("mt fuji", "Japan", "Tokyo"),
    ("fuji", "Japan", "Tokyo"),
    ("matterhorn", "Switzerland", "Zurich"),
    ("mont blanc", "France", "Geneva"),
    ("dolomites", "Italy", "Venice"),
    ("canadian rockies", "Canada", "Calgary"),
    ("mount rainier", "United States", "Seattle"),
    ("rainier", "United States", "Seattle"),
    ("mount shasta", "United States", "San Francisco"),
    ("mt shasta", "United States", "San Francisco"),
    ("klamath national forest", "United States", "San Francisco"),
]


def _canonical_country_main_v14(value: Any) -> str:
    raw = _clean_destination_piece(value)
    if not raw:
        return ""
    norm = _norm_key(raw)
    if not norm or re.search(r"\d", norm) or len(norm) <= 2:
        return ""
    if norm in _COUNTRY_ALIASES_MAIN_V14:
        return _COUNTRY_ALIASES_MAIN_V14[norm]
    if _pycountry_main_v14 is not None:
        try:
            found = _pycountry_main_v14.countries.lookup(raw)
            return getattr(found, "common_name", None) or getattr(found, "name", raw)
        except Exception:
            pass
    # Accept only country-looking fallback values, not state/address fragments.
    bad_fragments = {"california", "washington", "colorado", "nevada", "alberta", "british columbia"}
    if norm in bad_fragments:
        return ""
    words = norm.split()
    if 1 <= len(words) <= 4 and all(len(w) >= 3 for w in words):
        return raw
    return ""


def _place_hint_main_v14(destination: Any) -> Dict[str, str]:
    if isinstance(destination, dict):
        text = " ".join(str(destination.get(k, "")) for k in ("name", "city", "destination_name", "formatted_address"))
    else:
        text = str(destination or "")
    norm = _norm_key(text)
    for needle, country, flight_city in _PLACE_HINTS_MAIN_V14:
        if needle in norm:
            return {"country": country, "flight_search_city": flight_city}
    return {}


def _country_from_formatted_address(address: str) -> str:
    parts = [_clean_destination_piece(p) for p in str(address or "").split(",") if _clean_destination_piece(p)]
    for part in reversed(parts):
        country = _canonical_country_main_v14(part)
        if country:
            return country
    return ""


def _destination_country(destination: Any) -> str:
    hint = _place_hint_main_v14(destination)
    if hint.get("country"):
        return hint["country"]

    if isinstance(destination, dict):
        address_country = _country_from_formatted_address(str(destination.get("formatted_address") or ""))
        if address_country:
            return address_country
        for key in ("flight_search_country", "country"):
            value = _canonical_country_main_v14(destination.get(key))
            if value:
                return value
    return ""


def _destination_to_flight_query(destination: Any) -> str:
    hint = _place_hint_main_v14(destination)
    if hint.get("flight_search_city"):
        return hint["flight_search_city"]

    if isinstance(destination, str):
        return _clean_destination_piece(destination)
    if not isinstance(destination, dict):
        return _clean_destination_piece(destination)

    explicit = _clean_destination_piece(destination.get("flight_search_city"))
    if explicit and _norm_key(explicit) not in _BAD_CITY_MAIN_V14 and not re.search(r"\d", explicit):
        return explicit

    address = _clean_destination_piece(destination.get("formatted_address"))
    city_from_address = _city_from_formatted_address(address)
    if city_from_address and _norm_key(city_from_address) not in _BAD_CITY_MAIN_V14 and not re.search(r"\d", city_from_address):
        return city_from_address

    for key in ("destination_name", "city", "name", "country"):
        value = _clean_destination_piece(destination.get(key))
        if value and _norm_key(value) not in _BAD_CITY_MAIN_V14:
            return value
    return ""


def _normalize_destinations_for_flights(destinations: List[Any], origin: str = "") -> Dict[str, Any]:
    flight_destinations: List[str] = []
    no_flight_needed: List[Dict[str, Any]] = []
    skipped_duplicates: List[Dict[str, Any]] = []
    mappings: List[Dict[str, Any]] = []

    seen_cities = set()
    seen_countries = set()
    dedupe_countries = os.getenv("DEDUPE_FLIGHT_COUNTRIES", "1").strip().lower() not in {"0", "false", "no"}

    for destination in destinations:
        flight_query = _destination_to_flight_query(destination)
        if not flight_query:
            continue

        display_name = _destination_display_name(destination)
        country = _destination_country(destination)
        city_key = _norm_key(flight_query)
        country_key = _norm_key(country)

        record = {
            "display_name": display_name,
            "flight_search_city": flight_query,
            "country": country,
            "original": destination,
        }

        if _same_place_name(origin, flight_query):
            no_flight_needed.append({
                **record,
                "status": "no_flight_needed",
                "message": "No flight needed",
                "reason": "Destination is the same as origin.",
            })
            mappings.append({**record, "action": "no_flight_needed_same_as_origin"})
            if dedupe_countries and country_key:
                seen_countries.add(country_key)
            continue

        if city_key in seen_cities:
            skipped_duplicates.append({**record, "reason": "duplicate_city"})
            mappings.append({**record, "action": "skipped_duplicate_city"})
            continue

        if dedupe_countries and country_key and country_key in seen_countries:
            skipped_duplicates.append({**record, "reason": "duplicate_country"})
            mappings.append({**record, "action": "skipped_duplicate_country"})
            continue

        seen_cities.add(city_key)
        if dedupe_countries and country_key:
            seen_countries.add(country_key)
        flight_destinations.append(flight_query)
        mappings.append({**record, "action": "flight_search"})

    return {
        "flight_destinations": flight_destinations,
        "no_flight_needed": no_flight_needed,
        "skipped_duplicates": skipped_duplicates,
        "mappings": mappings,
        "dedupe_countries_enabled": dedupe_countries,
    }

