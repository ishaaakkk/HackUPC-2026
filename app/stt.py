import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def transcribe_audio(audio_file) -> str:
    audio_bytes = audio_file.read()
    
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    response = model.generate_content([
        {
            "mime_type": "audio/webm",
            "data": audio_bytes
        },
        "Transcribe exactly what is said in this audio. Return only the transcription, nothing else."
    ])
    
    return response.text.strip()