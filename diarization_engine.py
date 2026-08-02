"""Speaker diarization engine — pyannote.audio + speaker-diarization-community-1.

This replaces the pitch/MFCC + fixed-k KMeans backend that used to live inline in
stillscript._diarize(). Diarization is shared infrastructure: BOTH Fast mode and
Accurate mode call it, so this module is deliberately independent of either engine.

═══════════════════════════════════════════════════════════════════════════
 WHY THE OLD BACKEND WAS REPLACED (masterplan 2.12)
═══════════════════════════════════════════════════════════════════════════
The old path computed ONE feature vector per Whisper segment, and Whisper's
long-form segments are ~30s (47 of 50 on the 25-minute test recording). A
speaker change inside a segment was therefore structurally unrepresentable.
Measured on the real failing audio: the spread of 3s sub-windows *within* a 30s
block was 3.64 against 1.28 *between* the averaged 30s vectors — a 2.85x ratio.
Averaging destroyed speaker identity before clustering ever ran. A confirmed
case: a 210s stretch containing four distinct voices (verified by ear) came out
as a single speaker.

Two things follow, and both matter for anyone changing this file:

  * Diarization must NOT be derived from Whisper's segment boundaries. pyannote
    does its own speaker-aware segmentation over the raw audio; Whisper's
    segments are only used afterwards, to attach text to the speaker timeline.
  * The number of speakers is NOT forced. The old code passed the UI's
    "Expected speakers" value straight into KMeans as n_clusters, so a
    4-speaker recording left on the default of 2 could not come out right. The
    pipeline estimates the count itself — measured 4/4 correct on the segment
    that used to collapse to 1. See `diarize()` on why max_speakers is accepted
    but not enforced.

═══════════════════════════════════════════════════════════════════════════
 AUDIO INPUT — in-memory, on purpose
═══════════════════════════════════════════════════════════════════════════
pyannote 4.x decodes audio through `torchcodec`, whose published wheel is linked
against CUDA (it needs libnvrtc) and will NOT load on the CPU-only install this
product ships. Rather than fight that, audio is decoded here with the ffmpeg
binary the app already bundles and handed to the pipeline as an in-memory
waveform — a path pyannote documents and supports ("Processing from memory").
This is a deliberate, load-bearing design choice, not an incidental workaround:
do not "simplify" it back to passing a file path.

═══════════════════════════════════════════════════════════════════════════
 MODEL DISTRIBUTION
═══════════════════════════════════════════════════════════════════════════
Upstream `pyannote/speaker-diarization-community-1` is GATED — every end user
would have to accept its conditions on their own HF account, which a shipped
desktop app cannot do. It is CC-BY-4.0, which permits redistribution with
attribution, so it is mirrored unmodified to the repo below and fetched from
there. Attribution lives in the mirror's README and in the app's Credits list.

Only ~33.7 MB, so this deliberately does NOT reuse accurate_model_download.py's
chunked/resumable machinery: that exists because the Accurate model is 5.75 GiB
and takes hours, where a failed fetch here costs seconds. Integrity is a plain
SHA-256 pin per file (see _EXPECTED_SHA256) rather than accurate_guard.py's
four-layer check — that guard exists because a *fine-tuned* merge can be
silently substituted by stock weights and still produce fluent, wrong output.
This is an off-the-shelf pretrained pipeline with no such failure mode; a
content hash fully covers "is this the file we tested".

Heavy dependencies (torch, pyannote.audio) are imported lazily inside the
functions, never at module import time, so merely importing this module can
never break app startup.
"""

import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger("stillscript.diarization")

# Mirror of pyannote/speaker-diarization-community-1 (CC-BY-4.0, unmodified).
REPO_ID = "DanieClar/stillscript-speaker-diarization-community-1"
REVISION = "1f89d99ad83201c3fb7f504be2c4c0b5e6135359"

MODEL_DIR_ENV_VAR = "STILLSCRIPT_DIARIZATION_MODEL_DIR"
_MANAGED_DIR_NAME = "diarization-community-1"

# Every file the pipeline needs, with the SHA-256 of the exact bytes this build
# was tested against. Verified after download; a mismatch deletes the file and
# raises rather than loading something we did not test.
_EXPECTED_SHA256 = {
    "config.yaml":
        "5ce2bfa9a938dc132cec1172592d65173cbb8f444ea1e4133f10f9391de155be",
    "segmentation/pytorch_model.bin":
        "7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e",
    "embedding/pytorch_model.bin":
        "6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929",
    "plda/plda.npz":
        "9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255",
    "plda/xvec_transform.npz":
        "325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f",
}

SAMPLE_RATE = 16000

# Turns shorter than this are dropped from the speaker timeline before any text
# is attached. At 0.2s a turn cannot hold an intelligible word, and keeping them
# splits real sentences mid-phrase across speakers — actively worse than letting
# the surrounding speaker absorb the gap. community-1 produced 23 of these on
# the 210s test segment. Dropping, not merging: a dropped turn leaves a gap that
# the max-overlap assignment below fills naturally, whereas "merging" would have
# to invent a speaker for it.
MIN_TURN_SEC = 0.2

_pipeline_cache = {}


class DiarizationUnavailable(RuntimeError):
    """Diarization cannot run — dependencies or model files are missing.

    Raised rather than silently degrading. The caller decides whether to fall
    back, and is expected to make that visible rather than quietly shipping a
    one-speaker transcript (see the masterplan waitlist entry on silent
    degradation).
    """


def resolve_model_dir(model_dir=None):
    """Where the diarization model should be loaded from.

    Explicit argument, then the environment override, then a directory bundled
    beside the frozen executable (so an installer can ship the model and skip
    the download entirely), then the managed download location.
    """
    if model_dir:
        return str(model_dir)
    env = os.environ.get(MODEL_DIR_ENV_VAR)
    if env:
        return env

    # Bundled-with-the-app location, if an installer put it there.
    import sys
    if getattr(sys, "frozen", False):
        bundled = Path(sys.executable).parent / _MANAGED_DIR_NAME
        if (bundled / "config.yaml").is_file():
            return str(bundled)

    return str(Path.home() / ".stillscript_models" / _MANAGED_DIR_NAME)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_model_dir(model_dir):
    """Return (ok, detail). Checks every pinned file's SHA-256."""
    root = Path(model_dir)
    for rel, expected in _EXPECTED_SHA256.items():
        p = root / rel
        if not p.is_file():
            return False, f"missing file: {rel}"
        actual = _sha256(p)
        if actual != expected:
            return False, f"checksum mismatch for {rel}: {actual[:12]}... != {expected[:12]}..."
    return True, str(root)


def is_model_ready(model_dir=None):
    """True when ensure_model() would return without touching the network."""
    ok, _ = verify_model_dir(resolve_model_dir(model_dir))
    return ok


def ensure_model(model_dir=None, progress_callback=None):
    """Make sure the diarization model is on disk and verified; return its dir.

    Safe to call every run: when the files are present and their hashes match,
    this returns immediately with no network access.
    """
    target = Path(resolve_model_dir(model_dir))

    ok, detail = verify_model_dir(target)
    if ok:
        logger.info("Diarization model present and verified at %s", target)
        return str(target)

    logger.info("Fetching diarization model (%s) -> %s [%s]", REPO_ID, target, detail)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise DiarizationUnavailable(
            f"Speaker identification needs the 'huggingface_hub' package: {e}"
        ) from e

    target.mkdir(parents=True, exist_ok=True)
    try:
        for i, rel in enumerate(_EXPECTED_SHA256, start=1):
            if progress_callback:
                try:
                    progress_callback(rel, i, len(_EXPECTED_SHA256))
                except Exception:  # a UI bug must not kill the download
                    logger.warning("diarization progress hook raised", exc_info=True)
            hf_hub_download(
                repo_id=REPO_ID, revision=REVISION, filename=rel,
                local_dir=str(target),
            )
    except Exception as e:
        raise DiarizationUnavailable(
            "Could not download the speaker-identification model. This usually "
            f"means there is no internet connection right now. ({e})"
        ) from e

    ok, detail = verify_model_dir(target)
    if not ok:
        # Do not keep bytes we could not verify.
        for rel in _EXPECTED_SHA256:
            try:
                (target / rel).unlink()
            except OSError:
                pass
        raise DiarizationUnavailable(
            f"The downloaded speaker-identification model failed its integrity "
            f"check ({detail}). Please try again."
        )
    logger.info("Diarization model downloaded and verified at %s", target)
    return str(target)


def load_pipeline(model_dir=None):
    """Load (and cache) the pyannote pipeline, pinned to CPU."""
    resolved = ensure_model(model_dir)
    if resolved in _pipeline_cache:
        return _pipeline_cache[resolved]

    try:
        import torch
        import torch.torch_version
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise DiarizationUnavailable(
            f"Speaker identification needs the 'pyannote.audio' and 'torch' packages: {e}"
        ) from e

    # torch >=2.6 defaults weights_only=True. Allowlist only the benign metadata
    # classes pyannote's checkpoints carry -- deliberately narrower than
    # weights_only=False, which would permit arbitrary code execution on load.
    safe = [torch.torch_version.TorchVersion]
    try:
        from pyannote.audio.core.task import Specifications, Problem, Resolution
        safe += [Specifications, Problem, Resolution]
    except Exception:  # pragma: no cover - layout differs across pyannote versions
        logger.debug("pyannote task classes not importable for safe-globals", exc_info=True)
    torch.serialization.add_safe_globals(safe)

    logger.info("Loading diarization pipeline from %s", resolved)
    pipeline = Pipeline.from_pretrained(resolved)
    if pipeline is None:
        raise DiarizationUnavailable(
            f"pyannote could not build a pipeline from {resolved}."
        )
    pipeline.to(torch.device("cpu"))   # CPU only: this product ships CPU-only torch

    _pipeline_cache[resolved] = pipeline
    return pipeline


def _load_waveform(audio_path):
    """Decode `audio_path` to an in-memory 16 kHz mono waveform tensor.

    Uses the bundled ffmpeg (same binary the rest of the app relies on) rather
    than pyannote's own decoding, which goes through torchcodec — see the module
    docstring for why that cannot be used here.
    """
    import subprocess
    import tempfile
    import torch
    import soundfile as sf

    fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="stillscript_diar_")
    os.close(fd)
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", str(audio_path),
             "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", tmp_wav],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        data, sr = sf.read(tmp_wav, dtype="float32", always_2d=True)
    finally:
        try:
            os.remove(tmp_wav)
        except OSError:
            pass
    return {"waveform": torch.from_numpy(data.T), "sample_rate": sr}


def diarize(audio_path, max_speakers=None, model_dir=None):
    """Return the speaker timeline for `audio_path` as [(start, end, speaker)].

    Non-overlapping and sorted by start time.

    `max_speakers` is accepted for interface compatibility with the UI's
    "Expected speakers" control but is deliberately NOT passed to the pipeline.
    That control was built for the old KMeans backend, which needed an exact k;
    its default is 2, so honouring it would cap a 4-speaker recording at two
    speakers and reproduce the exact failure this backend replaced. The
    pipeline's own estimate was measured 4/4 correct on that recording. Changing
    the control's meaning is a UI decision, tracked separately in the masterplan.
    """
    if max_speakers is not None:
        logger.info("Ignoring max_speakers=%s; the pipeline estimates the "
                    "speaker count itself (masterplan 2.12).", max_speakers)

    pipeline = load_pipeline(model_dir)
    audio = _load_waveform(audio_path)
    output = pipeline(audio)

    # community-1 offers an "exclusive" (non-overlapping) diarization built
    # specifically to reconcile speaker turns with imprecise transcription
    # timestamps -- exactly this use case. Prefer it; fall back to the regular
    # annotation on any pipeline that does not provide it.
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", output)

    turns = [
        (float(seg.start), float(seg.end), str(spk))
        for seg, _, spk in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t[0])

    kept = [t for t in turns if (t[1] - t[0]) >= MIN_TURN_SEC]
    dropped = len(turns) - len(kept)
    if dropped:
        logger.info("Dropped %d speaker turn(s) shorter than %.2fs", dropped, MIN_TURN_SEC)

    logger.info("Diarization found %d speaker(s) across %d turn(s)",
                len({t[2] for t in kept}), len(kept))
    return kept
