from google import genai
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

# We use the same Gemini client as in llm.py
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def transcribe_audio(audio_file, language_code: str = "es") -> str:
    """
    Transcribes audio using Gemini 1.5 Flash (or 2.0/2.5 as configured).
    This replaces the ElevenLabs implementation.
    """
    # Create a temporary file to save the uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    uploaded_file = None
    try:
        # Upload to Gemini
        uploaded_file = client.files.upload(file=tmp_path, config={'mime_type': 'audio/webm'})
        
        # Generate transcription
        # We use flash as it is faster and cheaper
        model_id = os.getenv("GEMINI_MODEL_ID", "gemini-1.5-flash")
        
        response = client.models.generate_content(
            model=model_id,
            contents=[
                f"Transcribe this audio exactly as heard in {language_code}. Output ONLY the transcript text.",
                uploaded_file
            ],
        )
        return response.text.strip()
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass