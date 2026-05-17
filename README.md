# 🎙️ Audio Customer Support Agent (with RAG)

A modular, real-time audio-based customer support agent pipeline utilizing state-of-the-art **Speech-to-Text (STT)**, a **Large Language Model (LLM) ReAct Agent** enriched with **Retrieval-Augmented Generation (RAG)**, and high-fidelity **Text-to-Speech (TTS)**.

---

## 🚀 System Architecture & Flow

The pipeline orchestrates user interaction dynamically through modular voice processing steps:

```
[🎙️ Audio Input] ➔ [🗣️ STT (Speech-to-Text)] ➔ [🧠 LLM ReAct Agent]
                                                       🗂️ RAG (ChromaDB Query)
                                                               ⬇
[🔊 Audio Output] ➔ [📣 TTS (Text-to-Speech)] ➔ [📝 Response Text]
```

1.  **Audio Input:** User records/uploads speech via the web interface.
2.  **Speech-to-Text (STT):** Translates speech to text using **Deepgram API** or **OpenAI Whisper**.
3.  **LLM Customer Agent:** Formulates responses using the **LangChain ReAct Agent** framework.
    *   *Retrieval-Augmented Generation (RAG):* Performs semantic search using a local **ChromaDB** containing 16 comprehensive customer support documents embedded with **SentenceTransformers**.
4.  **Text-to-Speech (TTS):** Synthesizes response text back to voice audio bytes using **ElevenLabs** or free Microsoft neural voices (**Edge TTS**).

---

## ✨ Features

*   **Offline Fallback Capability:** Extremely resilient; if cloud API keys (e.g., OpenAI, ElevenLabs) are unavailable, the agent falls back instantly to a direct local **ChromaDB vector search** combined with free **Edge TTS** and Whisper/Mock voice layers.
*   **Detailed Analytics:** Real-time metrics breakdown showing elapsed duration for STT, LLM inference, TTS generation, and overall system processing time (in milliseconds).
*   **Dual UI Interface:** Fully-featured Streamlit UI with dedicated tabs for:
    *   `Text Chat` — Lightweight, text-only conversation queries.
    *   `Audio Chat` — Voice recording, playback, and full pipeline execution.
    *   `Health Monitor` — Real-time tracking of backend status and system components.
    *   `Documentation` — Built-in instruction manual and REST endpoint descriptions.

---

## 📁 Repository Structure

```
audio-support-agent/
├── .gitignore                      # Prevents committing credentials/virtualenvs
├── README.md                       # Main landing page & system overview
├── mid_session_requirements/       # Testing requirements & schemas
│   └── example_responses.json
└── audio_support_agent/            # Core application codebase
    ├── src/
    │   ├── api/
    │   │   └── server.py           # FastAPI REST Endpoint backend
    │   ├── llm/
    │   │   └── agent.py            # LangChain ReAct agent + ChromaDB RAG
    │   ├── stt/
    │   │   └── base_stt.py         # STT client (Deepgram / Whisper)
    │   ├── tts/
    │   │   └── base_tts.py         # TTS client (ElevenLabs / Edge TTS)
    │   ├── utils/
    │   │   └── kb_test.py          # Standalone ChromaDB ingestion tester
    │   └── pipeline.py             # Pipeline orchestrator (STT ➔ LLM ➔ TTS)
    ├── streamlit_app.py            # Streamlit testing dashboard
    ├── test_endpoints.py           # Endpoint integration test suite
    ├── requirements.txt            # Project dependencies
    ├── .env.example                # Template for service keys
    └── tests/                      # Unit testing suite
        └── test_stt.py
```

---

## ⚡ Quick Start

### 1. Installation & Environment Setup
Clone the repository and set up a Python virtual environment:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install core and dev dependencies
pip install -r requirements.txt
```

### 2. Configure Service Credentials
Copy the environment template and fill in your keys:
```bash
cp .env.example .env
```
*You can configure **Deepgram**, **ElevenLabs**, and **OpenAI** API keys. If left blank, the pipeline will gracefully switch to offline mocks and free Microsoft Edge TTS neural voices.*

---

## 🛠️ Running the Application

This system uses a decoupled frontend-backend architecture:

### Step 1: Start the API Backend
Launch the FastAPI uvicorn server:
```bash
cd audio_support_agent
python -m src.api.server
```
*By default, the server runs on `http://localhost:8000`. You can override the port by supplying `--port <number>`.*

### Step 2: Launch the Web UI Dashboard
In a separate terminal, launch the Streamlit frontend:
```bash
cd audio_support_agent
streamlit run streamlit_app.py
```
*The interactive dashboard will open automatically in your browser at `http://localhost:8501`.*

---

## 🧪 Testing & Verification

### Standalone Knowledge Base Test
You can run a local test to verify ChromaDB collection ingestion and document querying:
```bash
python src/utils/kb_test.py
```

### Automated Unit & Endpoint Testing
Run the comprehensive `pytest` test suite:
```bash
pytest tests/
```

Test REST API endpoints against the running uvicorn server:
```bash
python test_endpoints.py
```
