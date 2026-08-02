#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  StillScript Confidential Transcripts — Linux Installer (Ubuntu / Linux Mint)
#  Outeur: Danie
# ─────────────────────────────────────────────────────────────

set -e

APP_NAME="StillScript"
APP_DIR="$HOME/.local/share/stillscript"
# Masterplan 4.1: the pre-rename install location. Removed at the end of a
# successful install so an old launcher/desktop entry cannot keep pointing at
# a stale copy of the app.
LEGACY_APP_DIR="$HOME/.local/share/danscribe-ai"
BIN_DIR="$HOME/.local/bin"
ICON_DIR="$HOME/.local/share/icons"
DESKTOP_DIR="$HOME/.local/share/applications"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║      StillScript — Linux Installer           ║"
echo "║   Made in South Africa · for South Africans  ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Stap 1: Kontroleer Python 3 ──
echo "► Checking Python 3..."
if ! command -v python3 &>/dev/null; then
    echo "  Python 3 not found. Installing..."
    sudo apt-get update -qq
    sudo apt-get install -y python3 python3-pip python3-venv
else
    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "  Python $PY_VER found ✓"
fi

# ── Stap 2: Kontroleer pip ──
echo "► Checking pip..."
if ! python3 -m pip --version &>/dev/null; then
    sudo apt-get install -y python3-pip
fi
echo "  pip found ✓"

# ── Stap 3: Installeer stelselafhanklikhede ──
echo "► Installing system dependencies..."
sudo apt-get install -y \
    python3-tk \
    python3-venv \
    ffmpeg \
    libportaudio2 \
    libsndfile1 \
    2>/dev/null || true
echo "  System dependencies installed ✓"

# ── Stap 4: Skep app-gids ──
echo "► Creating application directory..."
mkdir -p "$APP_DIR"
mkdir -p "$BIN_DIR"
mkdir -p "$ICON_DIR"
mkdir -p "$DESKTOP_DIR"

# ── Stap 5: Kopieer app-lêers ──
echo "► Copying application files..."
cp "$SCRIPT_DIR/stillscript.py" "$APP_DIR/stillscript.py"
cp "$SCRIPT_DIR/accurate_engine.py" "$APP_DIR/accurate_engine.py"
cp "$SCRIPT_DIR/requirements.txt" "$APP_DIR/requirements.txt"

# Kopieer logo as dit bestaan
if [ -f "$SCRIPT_DIR/logo.jpg" ]; then
    cp "$SCRIPT_DIR/logo.jpg" "$APP_DIR/logo.jpg"
fi

# Kopieer dokumentasie
for f in DanScribe_UserGuide_v2.pdf DanScribe_ReleaseNotes_v2.pdf README.md; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
        cp "$SCRIPT_DIR/$f" "$APP_DIR/$f"
        echo "  Copied $f ✓"
    fi
done

# ── Stap 6: Skep 'n geïsoleerde virtuele omgewing ──
# Modern Debian/Ubuntu (PEP 668) blocks "pip install" against the system
# Python. A venv is never "externally managed", so this sidesteps that
# restriction cleanly instead of forcing it with --break-system-packages.
echo "► Setting up an isolated Python environment..."
if [ ! -d "$APP_DIR/venv" ]; then
    python3 -m venv "$APP_DIR/venv"
fi
echo "  Virtual environment ready ✓"

echo "► Installing Python packages (this may take a few minutes)..."
"$APP_DIR/venv/bin/pip" install --upgrade pip --quiet

# Install the CPU-only PyTorch build first. PyPI's default "torch" package
# bundles the full NVIDIA CUDA toolkit (several extra GB), which almost no
# laptop needs — Whisper's base/small models run fine on CPU. Installing
# this first satisfies openai-whisper's torch dependency below, so pip
# won't pull the much larger CUDA build afterward.
echo "  Installing PyTorch (CPU-only build)..."
"$APP_DIR/venv/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu

"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt"
echo "  Python packages installed ✓"

# ── Stap 7: Skep uitvoerbare skrip ──
echo "► Creating launcher script..."
cat > "$BIN_DIR/stillscript" << LAUNCHER
#!/bin/bash
cd "$APP_DIR"
"$APP_DIR/venv/bin/python3" "$APP_DIR/stillscript.py" "\$@"
LAUNCHER
chmod +x "$BIN_DIR/stillscript"
echo "  Launcher created at $BIN_DIR/stillscript ✓"

# ── Stap 8: Skep .desktop lêer vir Toepassingsmenu ──
echo "► Creating desktop entry (Applications menu)..."

# Gebruik logo as ikoon as dit beskikbaar is, anders 'n standaard ikoon
if [ -f "$APP_DIR/logo.jpg" ]; then
    cp "$APP_DIR/logo.jpg" "$ICON_DIR/stillscript.jpg"
    ICON_PATH="$ICON_DIR/stillscript.jpg"
else
    ICON_PATH="audio-x-generic"
fi

cat > "$DESKTOP_DIR/stillscript.desktop" << DESKTOP
[Desktop Entry]
Version=2.0
Type=Application
Name=StillScript
Comment=Confidential transcripts — professional audio transcription
Exec=$BIN_DIR/stillscript
Icon=$ICON_PATH
Terminal=false
Categories=AudioVideo;Audio;Utility;
Keywords=transcription;audio;whisper;afrikaans;south africa;
StartupNotify=true
DESKTOP

chmod +x "$DESKTOP_DIR/stillscript.desktop"
echo "  Desktop entry created ✓"

# ── Stap 9: Voeg ~/.local/bin by PATH as dit nie daar is nie ──
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo "" >> "$HOME/.bashrc"
    echo "# Added by StillScript installer" >> "$HOME/.bashrc"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    echo "  Added ~/.local/bin to PATH in .bashrc ✓"
fi

# ── Stap 10: Ruim die pre-hernoem-installasie op (masterplan 4.1) ──
# Only after everything above succeeded (set -e guarantees that). The user's
# DATA is not here — config, logs, models and transcripts live in $HOME and
# are migrated by stillscript.py itself on first launch. This directory holds
# only a copy of the app code and its venv, so removing it strands nothing.
if [ -d "$LEGACY_APP_DIR" ]; then
    echo "► Removing the previous DanScribe installation directory..."
    rm -rf "$LEGACY_APP_DIR"
    rm -f "$BIN_DIR/danscribe-ai" "$DESKTOP_DIR/danscribe-ai.desktop" \
          "$ICON_DIR/danscribe-ai.jpg"
    echo "  Old install removed (your settings, models and transcripts are kept) ✓"
fi

# ── Klaar ──
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║           Installation Complete! ✓           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  You can now launch StillScript:"
echo "  • From the Applications menu (search 'StillScript')"
echo "  • Or from the terminal: stillscript"
echo ""
echo "  Output files are saved to:"
echo "  ~/Documents/StillScript_Transcriptions/"
echo ""
