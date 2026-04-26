from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from app.stt import transcribe_audio
from app.llm import refine_locations_with_voice
import json
import requests as http_requests

router = APIRouter(prefix="/api")


import tempfile
import urllib.request
import os

# ---------- PHASE 1: Media Analysis ----------

@router.post("/analyze-media")
async def analyze_media(
    media: UploadFile = File(None),
    url: str = Form(None)
):
    """Receives an image/video file or URL, analyzes it with Gemini multimodal, returns candidate locations."""
    if not media and not url:
        raise HTTPException(status_code=400, detail="Must provide either media file or url")

    tmp_file = None
    try:
        # Create a temporary file
        fd, tmp_path = tempfile.mkstemp()
        os.close(fd)
        tmp_file = tmp_path

        mime_type = "image/jpeg"
        if media:
            content = await media.read()
            with open(tmp_path, "wb") as f:
                f.write(content)
            if media.content_type:
                mime_type = media.content_type
        elif url:
            # Download URL
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                content = response.read()
            with open(tmp_path, "wb") as f:
                f.write(content)
            # guess mime from URL extending
            if url.lower().endswith(".mp4"): mime_type = "video/mp4"
            elif url.lower().endswith(".png"): mime_type = "image/png"
            elif url.lower().endswith(".webm"): mime_type = "video/webm"
            else: mime_type = "image/jpeg"

        # Pass file path to the LLM function
        from app.llm import analyze_media_for_locations
        result = analyze_media_for_locations(tmp_path, mime_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing media: {str(e)}")
    finally:
        if tmp_file and os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except:
                pass


@router.get("/detect-origin")
async def detect_origin(request: Request):
    """Detects the user's origin city based on their IP address."""
    try:
        # Get client IP
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.headers.get("x-real-ip") or request.client.host

        # Clean IPv6 mapped IPv4
        if client_ip and client_ip.startswith("::ffff:"):
            client_ip = client_ip[7:]

        # Check if local IP
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
                "note": "IP local detectada, usando Barcelona por defecto"
            }

        # Lookup with ipapi.co
        resp = http_requests.get(
            f"https://ipapi.co/{client_ip}/json/",
            headers={"User-Agent": "HackUPC-Travel-App/1.0"},
            timeout=5
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
                "note": "No se pudo detectar la ubicación, usando Barcelona por defecto"
            }

        return {
            "status": "ok",
            "city": data.get("city", "Unknown"),
            "country": data.get("country_name", "Unknown"),
            "country_code": data.get("country_code", ""),
            "latitude": data.get("latitude", 0),
            "longitude": data.get("longitude", 0)
        }

    except Exception as e:
        return {
            "status": "error",
            "city": "Barcelona",
            "country": "Spain",
            "country_code": "ES",
            "latitude": 41.3874,
            "longitude": 2.1686,
            "note": f"Error: {str(e)}"
        }


# ---------- PHASE 2: Voice Validation ----------

@router.post("/voice-validate")
async def voice_validate(
    audio: UploadFile = File(...),
    locations: str = Form(...)
):
    """
    Receives audio + current locations JSON.
    Transcribes audio, then asks Gemini to refine/correct the locations.
    """
    try:
        locations_list = json.loads(locations)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="'locations' is not valid JSON")

    try:
        # Transcribe audio with ElevenLabs
        transcript = transcribe_audio(audio.file)

        # Refine locations with Gemini
        result = refine_locations_with_voice(locations_list, transcript)

        return {
            "transcript": transcript,
            "locations": result.get("locations", locations_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing voice: {str(e)}")


# ---------- PHASE 3: Flight Search ----------

@router.get("/search-flights")
def search_flights(origin: str, destinations: str, date: str = "2026"):
    """Searches for the cheapest flights from origin to each destination."""
    from flights import SkyscannerOptimizer
    from hotels import HotelSearcher
    import os

    api_key = os.getenv("SKYSCANNER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="SKYSCANNER_API_KEY not configured")

    optimizer = SkyscannerOptimizer(api_key)
    hotel_searcher = HotelSearcher(api_key)

    dest_list = [d.strip() for d in destinations.split(",") if d.strip()]

    try:
        results = optimizer.optimize_route([origin] + dest_list, date)

        # Add hotel prices
        for dest_name in results["results"]:
            results["results"][dest_name]["hotel_price"] = hotel_searcher.get_hotel_prices(dest_name)

        return results
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching flights: {str(e)}")