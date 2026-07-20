# DanScribe AI — Professional Edition v2.0

**DanScribe AI** is a desktop transcription tool powered by OpenAI's Whisper engine. Built in South Africa, for South Africans. It converts audio recordings into text across all 11 official South African languages, with speaker identification and AI-powered meeting summaries via Claude.

---

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
- Three AI model sizes: Base (fast), Small (accurate), Medium (professional)
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

```bash
pyinstaller --onefile --windowed --name "DanScribe_v2" --add-data "logo.jpg;." --icon "danscribe.ico" DanScribe_v2.py
```

The compiled executable will be in the `dist/` folder.

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
