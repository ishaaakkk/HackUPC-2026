# View it, Visit it — Multi-Modal Travel App (HackUPC 2026)

**View it, Visit it** is a cutting-edge travel web application that utilizes **multi-modal Artificial Intelligence** and **voice validation** to help you discover your next destination.

The project is divided into three logical phases integrated into a premium **Single Page Application (SPA)** experience.

---

## 🌟 Key Features

### Phase 1: Image & Origin Analysis
**Vision & Metadata**

- Image/Video upload to identify likely destinations using **Gemini 2.5 Flash** and **Google Vision API**.
- Automatic detection of the user's origin city via IP geolocation.
- Resolution of points of interest (POI) to real coordinates using **Google Places API**.

### Phase 2: Validation & Correction
**Voice UI & Human-in-the-Loop**

- Modern interface to review AI-generated suggestions.
- Voice validation to correct or confirm destinations by speaking directly to the browser.
- Transcription via **Web Speech API** (Browser) with fallback to **Gemini 1.5 Flash**.

### Phase 3: Flight Dashboard & Interactive Map

- Real-time flight search via the **Skyscanner API**.
- Interactive map based on **Leaflet** with price markers and route arcs.
- Side panel with dynamic images from Wikipedia and destination details:
  - Real-time weather (via **Open-Meteo**)
  - Landscape info
  - Hotels
  - Destination overview

---

## 🛠️ Technology Stack

- **Backend:** FastAPI (Python 3.10+)
- **Frontend:** HTML5, CSS3 (Modern UI), JavaScript (Vanilla ES6+)
- **AI/ML:** 
  - Google Gemini 2.5 Flash (Multi-modal analysis)
  - Google Cloud Vision (Landmark detection and OCR)
  - Web Speech API & Google Gemini (Speech-to-Text)
- **Maps:** Leaflet.js
- **Data APIs:**
  - Skyscanner (Flights)
  - Google Places (Geolocation and Photos)
  - Open-Meteo (Weather)
  - Wikipedia API (Images and content)

---

## 🚀 Quick Start Guide

---

## 1. Prerequisites and Demo Mode

You can try the application in two ways:

### A. Demo Mode (Without spending tokens)
Ideal for seeing how the site works without configuring all APIs. In this mode, pre-loaded (mock) results are used for image analysis and flight search, but **voice validation remains functional**.

To activate Demo Mode, add this variable to your `.env` or run it like this:
```bash
DEMO_MODE=1 python main.py
```

### B. Full Mode (Your own tokens)
To use the true power of AI, configure the following API Keys in your `.env` file:

| Service | Usage |
|---|---|
| Gemini API Key | Multi-modal image analysis, text refinement, and transcription fallback |
| (Optional) | Voice-to-text transcription uses the Public Web Speech API (Browser) |
| Skyscanner API Key | Real-time flight deals |
| Google Maps/Vision API Key | Landmark detection, place search, and maps |

---

## 2. Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ishaaakkk/HackUPC-2026.git
   cd HackUPC-2026
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

   *Note: If you plan to process video, ensure `opencv-python-headless` is installed.*

3. **Configure environment variables:**
   Create a `.env` file in the project root and add your credentials:
   ```env
   # API Keys (Optional if using DEMO_MODE=1)
   GEMINI_API_KEY=your_key_here
   SKYSCANNER_API_KEY=your_key_here
   GOOGLE_API_KEY=your_key_here
   ELEVENLABS_API_KEY=your_key_here

   # App Configuration
   DEMO_MODE=0           # Change to 1 to test without spending tokens
   LOCATION_FAST_MODE=1
   ```

---

## 3. Running the App

To start the server, simply run:

```bash
python main.py
```

The application will be running locally at: **http://127.0.0.1:8000** (or the address shown in your terminal).