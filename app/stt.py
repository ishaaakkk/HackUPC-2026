from google import genai
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def transcribe_audio(audio_file, language_code: str = "es") -> str:
    """
    Transcribes audio using Gemini 1.5 Flash.
    This replaces the ElevenLabs implementation.
    """
    client = get_client()
    if not client:
        return "[Error: GEMINI_API_KEY no configurada]"

    # Create a temporary file to save the uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    uploaded_file = None
    try:
        # Upload to Gemini
        uploaded_file = client.files.upload(file=tmp_path, config={'mime_type': 'audio/webm'})
        
        # Use a stable model name
        model_id = os.getenv("GEMINI_MODEL_ID", "gemini-1.5-flash")
        
        response = client.models.generate_content(
            model=model_id,
            contents=[
                f"Transcribe este audio exactamente como se escucha en {language_code}. Responde SOLO con el texto de la transcripción.",
                uploaded_file
            ],
        )
        return response.text.strip()
    except Exception as e:
        return f"[Error en transcripción: {str(e)}]"
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass