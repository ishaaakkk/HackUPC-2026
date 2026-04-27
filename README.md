# View it, Visit it — Multi-Modal Travel App (HackUPC 2026)

**View it, Visit it** es una aplicación web de viajes de vanguardia que utiliza **Inteligencia Artificial multimodal** y **validación por voz** para ayudarte a descubrir tu próximo destino.

El proyecto está dividido en tres fases lógicas integradas en una experiencia de **Single Page Application (SPA)** premium.

---

## 🌟 Características principales

### Fase 1: Análisis de Imagen y Origen
**Vision & Metadata**

- Subida de imagen para identificar destinos probables mediante **Gemini 2.5 Flash** y **Google Vision API**.
- Detección automática de la ciudad de origen del usuario mediante geolocalización por IP.
- Resolución de puntos de interés (POI) a coordenadas reales mediante **Google Places API**.

### Fase 2: Validación y Corrección
**Voice UI & Human-in-the-Loop**

- Interfaz moderna para revisar las sugerencias generadas por la IA.
- Validación por voz para corregir o confirmar destinos hablando directamente al navegador.
- Transcripción mediante **ElevenLabs Scribe** y refinamiento semántico con Gemini.

### Fase 3: Dashboard de Vuelos y Mapa Interactivo

- Búsqueda de vuelos en tiempo real mediante la **API de Skyscanner**.
- Mapa interactivo basado en **Leaflet** con marcadores de precio y arcos de ruta.
- Panel lateral con imágenes dinámicas de Wikipedia y detalles del destino:
  - Clima en tiempo real (vía **Open-Meteo**)
  - Paisajes
  - Hoteles
  - Información del lugar

---

## 🛠️ Stack Tecnológico

- **Backend:** FastAPI (Python 3.10+)
- **Frontend:** HTML5, CSS3 (Modern UI), JavaScript (Vanilla ES6+)
- **AI/ML:** 
  - Google Gemini 2.5 Flash (Análisis Multimodal)
  - Google Cloud Vision (Detección de Landmarks y OCR)
  - ElevenLabs (Speech-to-Text)
- **Mapas:** Leaflet.js
- **APIs de Datos:**
  - Skyscanner (Vuelos)
  - Google Places (Geolocalización y Fotos)
  - Open-Meteo (Clima)
  - Wikipedia API (Imágenes y contenido)

---

## 🚀 Guía de inicio rápido

---

## 1. Requisitos previos

Necesitarás configurar las siguientes API Keys en tu archivo `.env`:

| Servicio | Uso |
|---|---|
| Gemini API Key | Análisis multimodal de imágenes y refinamiento de texto |
| ElevenLabs API Key | Transcripción de voz a texto con Scribe v1 |
| Skyscanner API Key | Obtención de ofertas de vuelos reales |
| Google Maps/Vision API Key | Detección de lugares, búsqueda de sitios y mapas |

---

## 2. Instalación

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/tu-usuario/HackUPC2026.git
   cd HackUPC2026
   ```

2. **Crea un entorno virtual e instala las dependencias:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

   *Nota: Si vas a procesar video, asegúrate de tener `opencv-python-headless` instalado.*

3. **Configura las variables de entorno:**
   Crea un archivo `.env` en la raíz del proyecto y añade tus credenciales:
   ```env
   GEMINI_API_KEY=tu_clave_aqui
   SKYSCANNER_API_KEY=tu_clave_aqui
   GOOGLE_API_KEY=tu_clave_aqui
   ELEVENLABS_API_KEY=tu_clave_aqui
   LOCATION_FAST_MODE=1
   ```

---

## 3. Ejecución

Para iniciar el servidor de desarrollo, ejecuta:

```bash
uvicorn app.main:app --reload