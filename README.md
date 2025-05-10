# AI Lecture Transcription

**Cross-platform real-time transcription for live Microsoft Teams classes (not recorded by the host).**

---

## Features
- **Real-time system audio capture** (loopback) for Teams/Zoom/Meet lectures
- **Highly accurate transcription** using OpenAI Whisper (local) or Azure Speech-to-Text
- **Minimal live-caption UI** (Tkinter)
- **Session logging** to timestamped text files
- **Configurable via YAML** (audio device, language, VAD, etc.)
- **Summarization & keyword extraction** at session end
- **Testing & benchmarking** (WER, CER, RTF)
- **Cross-platform:** Windows & Macbook (see notes below)
- **One-click Windows packaging** (PyInstaller, optional code signing/installer)

---

## Quick Start

### 1. Install Python 3.8+

### 2. Clone the repository
```sh
git clone <repo-url>
cd AI_lecture_transcription
```

### 3. Install dependencies
```sh
pip install -r requirements.txt
```

### 4. Download NLTK data (first run only)
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 5. Configure your environment
- Copy `.env.example` to `.env` and fill in your Azure keys if using Azure STT.
- Edit `config.yml` for device, language, and model settings.

### 6. Run the application
```sh
python main.py
```

---

## Configuration

### `config.yml`
- **audio.device_name**: Name of loopback device (see `capture.py` test block for listing devices)
- **stt.engine**: `whisper` (local) or `azure` (cloud)
- **stt.language_hint**: e.g., `en-US`
- **stt.whisper.model_size**: e.g., `large-v3`, `base`, etc.
- **stt.azure.subscription_key/region**: Now referenced from `.env` (see below)
- **vad.threshold**: Energy threshold for speech detection
- **logging.log_directory**: Where transcripts are saved
- **ui**: Window size, font, etc.

### `.env`
```
AZURE_SPEECH_KEY=your_azure_key_here
AZURE_SPEECH_REGION=your_azure_region_here
```

**Note:** The application will load these values and inject them into the config at runtime.

---

## Packaging (Windows)

- Run the build script:
```sh
powershell -ExecutionPolicy Bypass -File build.ps1
```
- Produces a distributable folder in `dist_build/dist/LiveTranscriptionApp/`
- Optional: Enable code signing and installer creation in `build.ps1`

---

## Benchmarking & Testing

- Place `.wav` files in `tests/sample_audio/` and reference transcripts in `tests/reference_transcripts/`.
- Edit `tests/benchmark_config.yml` to list files and select STT engine.
- Run:
```sh
python tests/test_benchmark.py
```
- Reports WER, CER, and real-time factor (RTF).

---

## Extension Ideas
1. **Speaker Diarization**: Identify and label different speakers.
2. **Custom Vocabulary/Language Model Adaptation**: User-provided word lists or fine-tuning.
3. **Advanced VAD**: Integrate a dedicated VAD library for better silence/noise handling.
4. **Real-time Translation**: Live translation to other languages.
5. **Summarization & Keyword Extraction**: (Implemented) Generates a summary and keywords at session end.

---

## Platform Notes
- **Windows**: Loopback capture works with "Stereo Mix" or similar. If not available, install [VB-Cable](https://vb-audio.com/Cable/) or similar.
- **MacOS**: Use [BlackHole](https://github.com/ExistentialAudio/BlackHole) or [Loopback](https://rogueamoeba.com/loopback/) for system audio capture.
- **Linux**: Not officially supported, but may work with PulseAudio loopback.

---

## Troubleshooting
- **No audio device found**: Run `python capture.py` to list available devices.
- **Whisper errors about ffmpeg**: Install ffmpeg and ensure it's in your PATH.
- **Azure errors**: Ensure your `.env` is set and your Azure subscription is active.
- **NLTK errors**: Run the download command above to fetch required data.

---

## License
MIT 