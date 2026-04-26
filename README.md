# View it, Visit it — Multi-Modal Travel App (HackUPC 2026)

**View it, Visit it** es una aplicación web de viajes de vanguardia que utiliza **Inteligencia Artificial multimodal** y **validación por voz** para ayudarte a descubrir tu próximo destino.

El proyecto está dividido en tres fases lógicas integradas en una experiencia de **Single Page Application (SPA)** premium.

---

## 🌟 Características principales

### Fase 1: Análisis de Imagen y Origen  
**Vision & Metadata**

- Subida de imagen para identificar destinos probables mediante **Gemini 2.5 Flash**.
- Detección automática de la ciudad de origen del usuario mediante geolocalización por IP.

### Fase 2: Validación y Corrección  
**Voice UI & Human-in-the-Loop**

- Interfaz moderna para revisar las sugerencias generadas por la IA.
- Validación por voz para corregir o confirmar destinos hablando directamente al navegador.
- Transcripción mediante **ElevenLabs** y refinamiento posterior con LLM.

### Fase 3: Dashboard de Vuelos y Mapa Interactivo

- Búsqueda de vuelos en tiempo real mediante la **API de Skyscanner**.
- Mapa interactivo con marcadores de precio y arcos de ruta.
- Panel lateral con imágenes dinámicas de Wikipedia y detalles del destino:
  - Clima
  - Paisajes
  - Hoteles
  - Información del lugar

---

## 🚀 Guía de inicio rápido

Sigue estos pasos para ejecutar la aplicación en tu entorno local.

Compatible con:

- Linux
- macOS
- Windows

---

## 1. Requisitos previos

Necesitarás configurar las siguientes API Keys en tu archivo `.env`:

| Servicio | Uso |
|---|---|
| Gemini API Key | Análisis multimodal de imágenes y refinamiento de texto |
| ElevenLabs API Key | Transcripción de voz a texto con Scribe v1 |
| Skyscanner API Key | Obtención de ofertas de vuelos reales |

---

## 2. Instalación

Clona el repositorio y entra en la carpeta del proyecto:

```bash
cd HackUPC2026