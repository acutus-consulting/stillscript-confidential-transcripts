"""Accurate-mode transcription engine — Whisper large-v3 + Afrikaans adapter.

This is the engine behind "Akkuraat" (Accurate) mode. It is deliberately kept
in its own module, separate from the Fast (Whisper Medium) path in
stillscript.py, so the two engines cannot interfere with one another.

═══════════════════════════════════════════════════════════════════════════
 CRITICAL — DO NOT ROUTE THIS MODEL THROUGH CTranslate2 / faster-whisper
═══════════════════════════════════════════════════════════════════════════
Transcription here goes through HuggingFace `transformers` and a direct
`model.generate()` call, and nothing else.

The CT2 / faster-whisper conversion of this specific merged large-v3-Afrikaans
model is KNOWN BROKEN: it forces the language incorrectly and fabricates
fluent *English* text from Afrikaans audio — a silent, plausible-looking
wrong answer, which is the worst possible failure mode for a legal/clinical
transcript. The converted artefacts still exist on disk
(`~/whisper_afrikaans_spike/ct2_afrikaans_int8/`) but must not be used. Do not
"optimise" this module onto that path even though int8/CT2 looks faster.

═══════════════════════════════════════════════════════════════════════════
 Decode configuration (verified in Wave 0.5 on real multi-speaker Afrikaans)
═══════════════════════════════════════════════════════════════════════════
- Direct `generate()`, NOT `pipeline(chunk_length_s=...)`. The chunked
  pipeline hallucinated on long-form Afrikaans; the sequential, timestamp-
  driven long-form algorithm inside `generate()` is what was validated.
- `condition_on_prev_tokens=True` — measurably better than False on the
  Toets3 benchmark (fixed "onkoste"→"onkors" 4/4, and Eskom consistency),
  at ~22% slower. Two small non-spiralling stutters were observed; they are
  logged on the masterplan waitlist, not a blocker.
- Audio is fed whole (`truncation=False`, `return_attention_mask=True`) so
  `generate()` runs its own long-form windowing with the anti-hallucination
  heuristics (compression-ratio / logprob / no-speech thresholds + a
  temperature fallback ladder).

Expect roughly 3x real time on CPU: this is a batch / leave-it-running job,
not an interactive one.

Before the model is used for anything, load_engine() runs the adapter guard in
accurate_guard.py (masterplan 2.2), which proves the weights on disk really are
the fine-tuned Afrikaans merge and not stock large-v3 or a corrupt copy. It
costs ~1s, once per session. A guard failure raises AccurateModelGuardError and
must be allowed to propagate — never fall back to another engine.

Heavy dependencies (`torch`, `transformers`) are imported lazily *inside*
the functions below, never at module import time. Fast mode's runtime does
not ship `transformers`, and merely importing this module must never be able
to break the Fast path.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger("stillscript.accurate")

# ─────────────────────────────────────────────
#  MODEL LOCATION
# ─────────────────────────────────────────────
# The merged fp32 model produced by the spike's verify_and_merge.py (base
# large-v3 + André Oosthuizen's Afrikaans LoRA, merged via merge_and_unload).
# It is a standard HuggingFace model *directory* — from_pretrained() takes the
# directory, not a single weights file.
#
# Override with the STILLSCRIPT_ACCURATE_MODEL_DIR environment variable, e.g.
# once the model ships somewhere other than the spike directory.
_DEFAULT_MODEL_DIR = str(Path.home() / "whisper_afrikaans_spike" / "merged_afrikaans_fp32")
MODEL_DIR_ENV_VAR = "STILLSCRIPT_ACCURATE_MODEL_DIR"

# Files from_pretrained() needs before it will load anything.
_REQUIRED_MODEL_FILES = ("config.json", "model.safetensors", "preprocessor_config.json")

# Human-readable engine name, recorded in the provenance block alongside the
# model ID / revision / layout / guard fields _describe_model_provenance()
# adds (masterplan 2.6).
ACCURATE_ENGINE_LABEL = "StillScript Accurate — Whisper large-v3 + Afrikaans adapter"

# This engine is fine-tuned for Afrikaans. If a caller does not specify a
# language we use "af" rather than letting Whisper auto-detect, because
# auto-detect is the path that was never validated for this merge.
DEFAULT_LANGUAGE = "af"

# Verified Wave 0.5 decode settings.
CONDITION_ON_PREV_TOKENS = True
COMPRESSION_RATIO_THRESHOLD = 2.4
LOGPROB_THRESHOLD = -1.0
NO_SPEECH_THRESHOLD = 0.6

# Whisper's temperature fallback ladder: retry low-confidence segments at
# progressively higher temperatures. This is the configuration Wave 0.5 was
# verified with, so it is the default here.
#
# NOTE for masterplan item 2.9 (reproducibility choice): "Best effort" is this
# ladder; "Consistent" is plain 0.0 (deterministic, no sampling). That item
# only needs to pass a different `temperature=` value into transcribe() — no
# restructuring of this module required.
TEMPERATURE_BEST_EFFORT = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
TEMPERATURE_CONSISTENT = 0.0

# Masterplan 2.4: `transcribe()`'s `progress_callback` receives transformers'
# `monitor_progress` tensor, which reports feature-frame counts, not seconds or
# tokens. Verified empirically (real transcribe() call, real callback, logged
# every invocation) rather than assumed from the transformers docstring alone:
# both agree on 100 frames per second, tied to WhisperFeatureExtractor's fixed
# hop length, not something that varies by model size or audio content. Exposed
# here so the UI never has to hardcode a model-internal constant of its own.
PROGRESS_FRAMES_PER_SECOND = 100.0


def _describe_reproducibility(temperature):
    """Human label for the temperature value that actually decoded this
    transcript (masterplan 2.9) — derived from the real value `transcribe()`
    was called with, not re-derived from Settings at provenance time. This is
    the same "engine's result is the single source of truth" principle Wave
    2.6 already established for model_id/model_revision/etc.: if a future
    caller (a test, a script, a later feature) ever passes some other
    temperature, provenance must say so honestly rather than mislabelling it
    as one of the two named Settings choices it isn't.
    """
    if temperature == TEMPERATURE_CONSISTENT:
        return "Consistent"
    if temperature == TEMPERATURE_BEST_EFFORT:
        return "Best effort"
    return f"Custom (temperature={temperature!r})"

SAMPLE_RATE = 16000

# Cache keyed by resolved model directory, so a second transcription in the
# same session does not re-read 6 GB from disk.
# (Evicting this before/after a run — to keep Medium and large-v3 from being
# resident at the same time — is masterplan item 2.5, intentionally not here.)
_engine_cache = {}


# The Accurate-mode error hierarchy lives in accurate_guard so that the UI can
# import and handle these without pulling in torch/transformers. Re-exported
# here because this module is the public entry point for Accurate mode.
#   AccurateEngineUnavailable  — Accurate mode cannot run on this machine
#   AccurateModelGuardError    — ...specifically because the model is wrong
from accurate_guard import (  # noqa: F401  (re-export)
    AccurateEngineUnavailable,
    AccurateModelGuardError,
    verify_merged_model,
)


def resolve_model_dir(model_dir=None):
    """Return the directory the accurate model should be loaded from.

    Explicit argument wins, then the environment variable, then the default
    spike location.
    """
    return model_dir or os.environ.get(MODEL_DIR_ENV_VAR) or _DEFAULT_MODEL_DIR


def check_model_available(model_dir=None):
    """Return (ok, detail) describing whether the model directory looks usable.

    Cheap filesystem check only — it does not load the model. Useful for a
    pre-flight check before committing the user to a multi-hour run.
    """
    resolved = resolve_model_dir(model_dir)
    if not os.path.isdir(resolved):
        return False, f"Model directory not found: {resolved}"
    missing = [f for f in _REQUIRED_MODEL_FILES if not os.path.isfile(os.path.join(resolved, f))]
    if missing:
        return False, f"Model directory {resolved} is missing: {', '.join(missing)}"
    return True, resolved


def load_engine(model_dir=None):
    """Load (and cache) the processor + model.

    Returns (processor, model, dir, guard_report). guard_report is the result
    of the adapter-guard check below — cached alongside the model (masterplan
    2.6) so every transcription in a session, not just the first, can report
    which verification mode (probe/full) actually passed for the model it
    used.

    Raises AccurateEngineUnavailable if the dependencies or the model files
    are not present, rather than falling back to anything else.
    """
    resolved = resolve_model_dir(model_dir)
    if resolved in _engine_cache:
        processor, model, guard_report = _engine_cache[resolved]
        return processor, model, resolved, guard_report

    ok, detail = check_model_available(resolved)
    if not ok:
        raise AccurateEngineUnavailable(detail)

    # Adapter guard (masterplan 2.2) — prove this really is the fine-tuned
    # Afrikaans merge before we transcribe a word with it. Deliberately placed
    # BEFORE from_pretrained(): it reads ~40 MB of probe tensors straight from
    # the safetensors file, so a wrong or corrupt model fails in about a second
    # rather than after a 6.2 GB load. It runs once per model directory, on the
    # cache-miss path only, so it costs nothing per transcription.
    #
    # AccurateModelGuardError is allowed to propagate. Do NOT catch it and fall
    # back to another engine — the entire point is that a wrong model must stop
    # the run rather than quietly produce a plausible transcript.
    guard_report = verify_merged_model(resolved)
    logger.info("Accurate model guard: %s", guard_report)

    # Lazy imports: keep `transformers`/`torch` out of the Fast-mode runtime.
    try:
        import torch  # noqa: F401  (needed for dtype below)
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
    except ImportError as e:
        raise AccurateEngineUnavailable(
            f"Accurate mode needs the 'transformers' and 'torch' packages: {e}"
        ) from e

    logger.info("Loading accurate-mode model from %s", resolved)
    try:
        processor = WhisperProcessor.from_pretrained(resolved)
        model = WhisperForConditionalGeneration.from_pretrained(resolved, dtype=torch.float32)
    except (MemoryError, RuntimeError) as e:
        # Masterplan 2.5: Fast mode's model is released before this call, but
        # that only removes ~1.5 GB of headroom pressure — it does not
        # guarantee large-v3's own ~8.7 GB peak fits on a memory-constrained
        # machine. Nothing is left half-loaded here: _engine_cache is only
        # written after BOTH from_pretrained() calls succeed (below), so a
        # failure on either one leaves it exactly as it was.
        if _looks_like_out_of_memory(e):
            raise AccurateEngineUnavailable(
                "Accurate mode needs roughly 8-9 GB of free memory to load its "
                "language model, and this computer does not have enough "
                "available right now. Close other programs to free up memory "
                "and try again, or use Fast mode instead — it uses a much "
                "smaller model and is unaffected by this."
            ) from e
        # A RuntimeError that isn't shaped like OOM (a corrupt file, a
        # transformers/safetensors internal error, ...) is a different problem
        # with a different fix; let its own message through rather than
        # relabelling it as a memory issue it may not be.
        raise
    model.eval()

    _engine_cache[resolved] = (processor, model, guard_report)
    return processor, model, resolved, guard_report


def _looks_like_out_of_memory(exc):
    """Distinguish a real allocator-refused-to-allocate failure from any other
    RuntimeError from_pretrained() might raise (a corrupt file, a
    transformers-internal error, ...), so only the former gets rewritten into
    the plain-language message above — everything else keeps its own,
    unmodified message.

    MemoryError is Python's own signal for this. The RuntimeError text is
    matched against torch's actual CPU allocator failure, confirmed against a
    real allocation-refused error on this machine
    ("[enforce fail at alloc_cpu.cpp:...] ... DefaultCPUAllocator: can't
    allocate memory: ..."), plus the CUDA equivalent and the generic phrase
    PyTorch uses for both, in case a future build runs on a GPU machine.
    """
    if isinstance(exc, MemoryError):
        return True
    text = str(exc)
    return (
        "DefaultCPUAllocator" in text
        or "CUDA out of memory" in text
        or "out of memory" in text.lower()
    )


def transcribe(
    path,
    *,
    language=DEFAULT_LANGUAGE,
    task="transcribe",
    model_dir=None,
    temperature=TEMPERATURE_BEST_EFFORT,
    condition_on_prev_tokens=CONDITION_ON_PREV_TOKENS,
    progress_callback=None,
):
    """Transcribe `path` with the large-v3 Afrikaans engine.

    Returns the same shape the Fast seam returns, so everything downstream
    (diarization, transcript rendering, provenance) keeps working unchanged:

        {"text": str,
         "segments": [{"start": float, "end": float, "text": str}, ...],
         "language": str,
         "engine": str}

    `progress_callback` is optional and receives the raw progress tensor from
    transformers' `monitor_progress` hook, forwarded here unmodified as
    `generate_kwargs["monitor_progress"]`. Verified (2026, masterplan 2.4) —
    both by reading transformers' own source and by a real transcribe() call
    with a callback that logged every invocation — to behave as follows, none
    of which is obvious from the parameter name alone:

      * Shape `(n, 2)`, `n` = batch size — always 1 here, this engine never
        batches. `p[0, 0]` = the audio-frame index currently being decoded;
        `p[0, 1]` = the total frame count for this audio. Both are counts of
        100-Hz feature frames (PROGRESS_FRAMES_PER_SECOND above), i.e.
        `p[0, 0] / PROGRESS_FRAMES_PER_SECOND` is seconds of audio reached so
        far — NOT tokens generated, NOT wall-clock time, NOT a 0-1 fraction
        (callers divide the two themselves).
      * Called once per long-form decoding window (Whisper's fixed ~30s
        window), at the START of that window, before its temperature-fallback
        retries (if any) run — so it does NOT fire again mid-retry, and a
        slow/difficult window means a long real gap with no call at all.
        Measured real gaps between calls on one run: 6s, 64s, 66s wall-clock
        for three consecutive ~30s-audio windows — i.e. call spacing is tied
        to how long each window takes to decode, not to a fixed interval.
      * Never called at all for short-form audio (a single window, no
        seek-loop) — this engine's `truncation=False` + attention-mask setup
        forces the long-form path for any audio this product actually
        transcribes, but a caller passing a clip under Whisper's window
        length would see zero calls. Progress UI must tolerate that (and any
        other zero-calls case) as a normal outcome, not an error.

    The time-warning / progress UI itself is masterplan 2.4.
    """
    import torch
    import librosa

    processor, model, resolved, guard_report = load_engine(model_dir)

    if language is None:
        # Don't quietly auto-detect on an Afrikaans-specialised merge.
        logger.info("No language given for accurate mode; defaulting to '%s'.", DEFAULT_LANGUAGE)
        language = DEFAULT_LANGUAGE

    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    duration = len(audio) / SAMPLE_RATE
    logger.info(
        "Accurate transcription starting: %s (%.1fs audio, language=%s, task=%s, "
        "condition_on_prev_tokens=%s)",
        path, duration, language, task, condition_on_prev_tokens,
    )

    # truncation=False + attention mask => generate() takes the long-form
    # sequential path with its own windowing and fallback heuristics.
    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        truncation=False,
        padding="longest",
        return_attention_mask=True,
    )

    generate_kwargs = dict(
        return_timestamps=True,     # required for >30s audio
        return_segments=True,       # gives per-segment start/end for diarization
        language=language,
        task=task,
        condition_on_prev_tokens=condition_on_prev_tokens,
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        logprob_threshold=LOGPROB_THRESHOLD,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        temperature=temperature,
    )
    if progress_callback is not None:
        generate_kwargs["monitor_progress"] = progress_callback

    with torch.no_grad():
        output = model.generate(**inputs, **generate_kwargs)

    return _build_result(processor, output, language, duration, resolved, guard_report, temperature)


def _describe_model_provenance(resolved, guard_report):
    """What backs this specific transcription, for the audit footer
    (masterplan 2.6). Combines two DIFFERENT verification events, both real
    and both worth recording distinctly rather than conflated:

      * The download-time stamp (accurate_model_download.py), if `resolved`
        is a managed download directory — Waves 2.1a.2/2.1a.3 always verify
        in FULL mode at download time, unconditionally, regardless of this
        session's own guard mode. Absent for a directory the user pointed
        STILLSCRIPT_ACCURATE_MODEL_DIR at directly (never downloaded by this
        app, so never stamped) — in that case model_revision/model_layout
        come back None rather than a guessed value.
      * This SESSION's own load-time guard result (guard_report, from
        load_engine()'s verify_merged_model() call) — probe mode by default
        (masterplan 2.2a), full mode only if STILLSCRIPT_ACCURATE_FULL_VERIFY
        is set. This is the check that actually gated THIS load, so it is
        reported as "guard_verification" regardless of what the stamp says.

    accurate_model_download is imported lazily here, matching this module's
    own convention of keeping anything not needed for the core generate()
    path out of load time — read_stamp() itself is cheap (stdlib only), but
    there is no reason for accurate_engine to import a download-oriented
    module until a result actually needs describing.
    """
    if guard_report.get("full"):
        guard_label = (f"Full ({guard_report.get('tensors_hashed', '?')} of "
                       f"{guard_report.get('tensor_count', '?')} tensors hashed)")
    else:
        guard_label = f"Probe ({guard_report.get('probes_checked', '?')} sample tensors)"

    import accurate_model_download as amd
    info = amd.describe_verified_model(resolved)
    if info:
        return {
            "model_id": info["repo_id"],
            "model_revision": info["revision"],
            "model_layout": info["layout"],
            "guard_verification": guard_label,
        }
    return {
        "model_id": amd.REPO_ID,
        "model_revision": None,
        "model_layout": None,
        "guard_verification": guard_label,
    }


def _build_result(processor, output, language, duration, resolved, guard_report, temperature):
    """Normalise generate()'s output into the Fast seam's result shape."""
    # With return_segments=True, generate() returns a dict:
    #   {"sequences": tensor, "segments": [[seg, seg, ...]]}   (outer list = batch)
    sequences = output["sequences"] if isinstance(output, dict) else output
    text = processor.batch_decode(sequences, skip_special_tokens=True)[0]

    segments = []
    if isinstance(output, dict) and output.get("segments"):
        for seg in output["segments"][0]:
            seg_text = processor.decode(seg["tokens"], skip_special_tokens=True).strip()
            if not seg_text:
                continue
            segments.append({
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg_text,
            })

    if not segments:
        # Never hand downstream an empty segment list for non-empty audio —
        # diarization would silently produce a single-speaker transcript.
        # One whole-file segment is honest about what we actually know.
        logger.warning("Accurate engine returned no timestamped segments; "
                       "falling back to a single whole-file segment.")
        segments = [{"start": 0.0, "end": duration, "text": text.strip()}]

    return {
        "text": text,
        "segments": segments,
        "language": language,
        "engine": ACCURATE_ENGINE_LABEL,
        "reproducibility": _describe_reproducibility(temperature),
        **_describe_model_provenance(resolved, guard_report),
    }
