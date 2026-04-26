from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from app.stt import transcribe_audio
from app.llm import refine_locations_with_voice
import json
import requests as http_requests
import tempfile
import urllib.request
import os
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


# ---------- PHASE 1: Media Analysis ----------

@router.post("/analyze-media")
async def analyze_media(
    media: UploadFile = File(None),
    url: str = Form(None)
):
    """Receives an image/video file or URL and analyzes it with Vision + Places."""
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
                "note": "IP local detectada, usando Barcelona por defecto",
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
                "note": "No se pudo detectar la ubicación, usando Barcelona por defecto",
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
    audio: UploadFile = File(...),
    locations: str = Form(...)
):
    try:
        locations_list = json.loads(locations)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="'locations' is not valid JSON")

    try:
        transcript = transcribe_audio(audio.file)
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

    api_key = os.getenv("SKYSCANNER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="SKYSCANNER_API_KEY not configured")

    dest_list = _parse_destinations(destinations)
    if not dest_list:
        raise HTTPException(status_code=400, detail="No destinations provided")

    optimizer = SkyscannerOptimizer(api_key)
    hotel_searcher = HotelSearcher(api_key)

    try:
        results = optimizer.optimize_route([origin] + dest_list, date)
        for dest_name, info in results.get("results", {}).items():
            # Use flight_search_city for hotel price when available, otherwise destination name.
            hotel_key = info.get("flight_search_city") or info.get("destination_name") or dest_name
            info["hotel_price"] = hotel_searcher.get_hotel_prices(hotel_key)
        return results
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching flights: {str(e)}")
