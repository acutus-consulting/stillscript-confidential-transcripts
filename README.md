# DanScribe AI — Professional Edition v3.1.0

**DanScribe AI** is a desktop transcription tool powered by OpenAI's Whisper engine. Built in South Africa, for South Africans. It converts audio recordings into text across all 11 official South African languages, with speaker identification and AI-powered meeting summaries via Claude.

---

## What's New in v3.1.0

- **Two transcription modes** — the model picker is replaced by a **Fast (Vinnig)** / **Accurate (Akkuraat)** mode selector. Fast (the default) runs the Whisper Medium engine with the improved Afrikaans prompt. Accurate is present but not yet enabled (coming in a later release).
- **Transcript provenance** — every `.txt` and `.docx` now carries an audit footer recording which engine produced it (model + version, mode, language setting, task, and timestamp) — useful for professional and legal record-keeping.
- **Credits / About surface** — a new About dialog attributing the open-source work DanScribe builds on (OpenAI Whisper).
- **Lower memory on long recordings** — speaker identification on long files (20 min+) now streams audio from a temporary file instead of loading the whole recording into RAM, cutting peak memory on multi-hour recordings from gigabytes to roughly constant. Short files are unchanged.
- **Self-contained Windows build** — the Windows `.exe` now bundles ffmpeg, so there is no separate install step; CI produces a CPU-only build with all model/library assets included.

## What's New in v3.0.1

- **Reduced Afrikaans→Dutch spelling drift** — Whisper's Afrikaans transcription prompt is now a longer, natural passage (including "nie ... nie" double negation, a construction Dutch doesn't have) instead of a short meta-instruction, giving the model a much stronger signal toward Afrikaans over Dutch orthography. For best results with Afrikaans audio, the Medium model is also recommended over Base/Small.
- **Linux installer fixes** — resolved a `pip install` failure on modern Debian/Ubuntu-based systems (PEP 668 "externally-managed-environment") by installing into an isolated virtual environment; PyTorch now installs as a CPU-only build, cutting the download size dramatically for the vast majority of users without an NVIDIA GPU.
- **CI-built Windows releases** — Windows `.exe` builds are now produced automatically by GitHub Actions on every tagged release, since PyInstaller requires building on Windows itself.

## What's New in v3.0

- **Fixed a startup crash** — the app could fail to launch due to a stray reference to an undefined function.
- **Secure API key storage** — the Claude API key now lives in your OS keyring (Windows Credential Manager, macOS Keychain, Linux Secret Service) instead of a plaintext file; falls back to a permission-restricted file if no keyring is available.
- **AI Summary model updated** — now uses Claude Opus 4.8.
- **Diagnostic logging** — a log file (`~/.danscribe.log`) now captures errors and warnings for easier troubleshooting.
- **More robust error handling** — clearer messages for authentication failures, rate limits, and API errors; corrupted settings files no longer crash the app.
- **Thread-safety fixes** — background transcription/summary work no longer touches the UI directly, preventing rare crashes and freezes.
- **File validation** — the app now checks that a selected audio file actually exists and isn't empty before processing.
- **Added `requirements.txt`** for reproducible installs.
- **Repository cleanup** — removed unused legacy source files and fixed a `.gitignore` bug that let the local config file be tracked by git.

## What's New in v2.0

- **Speaker Identification** — Automatically detects and labels different speakers (Speaker 1, Speaker 2, etc.)
- **Assign Speaker Names** — Replace speaker labels with real names after transcription
- **AI Summary via Claude** — Generate professional meeting summaries and minutes with one click
- **Claude API Integration** — Enter your own Claude API key via the Settings window
- **English UI** — Full English interface throughout
- **SA Flag** — Made in South Africa branding in the UI
- **Model Caching** — Models load once and stay cached for the session
- **Auto-Detect default** — Language defaults to Auto-Detect on first run
- **Translate to English default** — Task defaults to Translate to English
- **Word + Text output** — All transcriptions and summaries saved as both `.txt` and `.docx`
- **Settings persistence** — Last used model and language remembered between sessions

---

## Features

- Transcribe audio in the original language or translate directly to English
- Supports all 11 official South African languages
- Two transcription modes: Fast (Vinnig) — Whisper Medium — with Accurate (Akkuraat) coming in a later release
- Speaker identification with configurable number of expected speakers
- AI-powered meeting summaries (requires Claude API key)
- Output saved automatically as `.txt` and `.docx` to your Downloads folder
- Clean dark-mode desktop interface

## Supported Audio Formats

`.mp3` `.wav` `.m4a` `.flac` `.ogg` `.mp4` `.webm`

## Supported Languages

| Code | Language   |
|------|------------|
| Auto | Auto-Detect |
| af   | Afrikaans  |
| en   | English    |
| st   | Sesotho    |
| zu   | isiZulu    |
| xh   | isiXhosa   |
| tn   | Setswana   |
| nso  | Sepedi     |
| ss   | Siswati    |
| ve   | Tshivenda  |
| ts   | Xitsonga   |
| nr   | isiNdebele |

## System Requirements

- Windows 10 or 11 (64-bit)
- Python 3.9 or higher (for running from source)
- RAM: 4 GB minimum — 8 GB recommended for Medium model
- Disk: 150 MB (Base) up to 1.5 GB (Medium)
- Internet: Required on first run to download AI model

## Dependencies (running from source)

```bash
pip install -r requirements.txt
```

## Running from Source

```bash
python DanScribe_v2.py
```

## Building the EXE

Windows builds are produced automatically by CI — see
`.github/workflows/build-windows-release.yml` (PyInstaller cannot
cross-compile, so the build runs on a Windows runner). It installs CPU-only
PyTorch, bundles a static ffmpeg, collects the Whisper/librosa assets, builds
a self-contained one-file exe, and attaches `DanScribe_v3.exe` to the matching
GitHub release. It runs on every pushed `vX.Y.Z` tag.

To build locally on a Windows machine, mirror that workflow's `pyinstaller`
invocation (CPU torch + static ffmpeg staged alongside, then the
`--collect-all`/`--add-binary` flags). The compiled executable lands in
`dist/` as `DanScribe_v3.exe`.

## Claude API Key

To use the AI Summary feature:
1. Register at [console.anthropic.com](https://console.anthropic.com)
2. Create an API key
3. Open DanScribe AI → ⚙️ Settings → paste your key

Your key is stored locally on your device only. When available, it is saved in
your operating system's secure keyring (Windows Credential Manager, macOS
Keychain, or the Linux Secret Service). If no keyring backend is present, it
falls back to a permission-restricted config file (`~/.danscribe_config.json`,
readable only by your user account).

## Output

All files are saved to:
```
C:\Users\YourName\Downloads\DanScribe_Transcriptions\
```

Each transcription produces:
- `filename_transcript_YYYYMMDD_HHMMSS.txt`
- `filename_transcript_YYYYMMDD_HHMMSS.docx`

Each summary produces:
- `summary_YYYYMMDD_HHMMSS.txt`
- `summary_YYYYMMDD_HHMMSS.docx`

---

*Made in South Africa · for South Africans*  
*Built with OpenAI Whisper + CustomTkinter + Anthropic Claude*
