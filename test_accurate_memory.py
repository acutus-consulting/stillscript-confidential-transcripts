"""Tests for masterplan 2.5 — releasing Fast mode's model before Accurate
mode loads large-v3, and recovering cleanly if that load fails.

WHAT IS AND IS NOT HERE
────────────────────────
The actual peak-memory PROOF for this wave is a set of real, standalone
subprocess runs (VmHWM — the kernel's own peak-RSS tracking, immune to
sampling gaps a polling loop could miss), not part of this file: each
scenario needs its own real ~6 GiB large-v3 load, which takes minutes, and
running four of those inside one pytest-style suite would make it far too
slow to actually get re-run. The real numbers are reported in the delivery
summary and masterplan 2.5's entry. What THIS file covers, fast and
repeatably, is the mechanical/functional correctness the memory numbers
depend on:

  * release_fast_mode_model() actually drops the last reference (proven via
    weakref, not just "the dict is empty") and is a genuine no-op when
    nothing is loaded.
  * transcribe_audio_accurate() calls it BEFORE delegating to the engine,
    every time — checked by call-order, with the engine call itself mocked
    so this runs in well under a second.
  * A load failure (OOM-shaped, simulated on a real from_pretrained() call
    rather than assumed) is turned into a clear AccurateEngineUnavailable
    message, and does NOT leave a half-populated _engine_cache.
  * A non-OOM RuntimeError from the same call site is left alone — the
    classifier must not relabel every failure as a memory problem.
  * After a real, simulated load failure, Fast mode still works — a REAL
    Medium load + real transcription, in the same process, right after the
    failure, not assumed to "probably be fine".

Run with the full-requirements venv (needs whisper + torch + transformers
together):
    <appvenv>/bin/python3 test_accurate_memory.py
"""

import gc
import logging
import os
import sys
import weakref

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

import stillscript as ds  # noqa: E402
import accurate_engine  # noqa: E402
from accurate_guard import AccurateEngineUnavailable  # noqa: E402

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


CLIP = os.path.expanduser("~/whisper_afrikaans_spike/bench_30s.wav")


# ════════════════════════════════════════════════════════════════════════════
print("=== 1. release_fast_mode_model() — genuine release, not just a dict pop ===")
# ════════════════════════════════════════════════════════════════════════════
check("no-op when nothing is loaded (not an error)",
      ds.release_fast_mode_model() is None and not ds._model_cache)

# A tiny real object standing in for a loaded model — this section is about
# proving the RELEASE mechanism (dict-clear -> refcount drop -> gc), not about
# re-loading a real multi-hundred-MB Whisper model just to prove that part.
class _FakeModel:
    pass


fake = _FakeModel()
ref = weakref.ref(fake)
ds._model_cache["medium"] = fake
del fake  # the cache is the only other reference now

check("weakref still alive while the cache holds the only reference",
      ref() is not None)

ds.release_fast_mode_model()
check("_model_cache is empty after release", ds._model_cache == {})
check("the object is GENUINELY collected — weakref is dead, not just removed "
      "from one dict while something else keeps it alive",
      ref() is None)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 2. transcribe_audio_accurate() releases Fast mode's model BEFORE "
      "calling the engine, every time ===")
# ════════════════════════════════════════════════════════════════════════════
call_order = []

real_release = ds.release_fast_mode_model
real_import_accurate_engine = None  # not needed; we patch the module attr directly

def spy_release():
    call_order.append("release")
    real_release()


def spy_transcribe(path, **kw):
    call_order.append("engine.transcribe")
    return {"text": "toets", "segments": [], "language": "af", "engine": "x"}


ds.release_fast_mode_model = spy_release
ds._model_cache["medium"] = object()  # pretend Fast mode is loaded
real_accurate_transcribe = accurate_engine.transcribe
accurate_engine.transcribe = spy_transcribe
try:
    ds.transcribe_audio_accurate(CLIP, language="af", task="transcribe", model_dir="/x")
    check("release_fast_mode_model() ran", "release" in call_order)
    check("...and ran BEFORE the engine call, not after or interleaved",
          call_order.index("release") < call_order.index("engine.transcribe"),
          call_order)
finally:
    ds.release_fast_mode_model = real_release
    accurate_engine.transcribe = real_accurate_transcribe
    ds._model_cache.clear()


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 3. Out-of-memory classification — real error text, not guessed ===")
# ════════════════════════════════════════════════════════════════════════════
# The exact text captured from a real torch CPU allocator refusal on this
# machine (torch.empty(10**15, dtype=torch.float32)), not invented.
REAL_CPU_OOM_TEXT = (
    "[enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: "
    "can't allocate memory: you tried to allocate 4000000000000000 bytes. "
    "Error code 12 (Cannot allocate memory)"
)

for label, exc, expected in (
    ("a real captured CPU allocator failure", RuntimeError(REAL_CPU_OOM_TEXT), True),
    ("a bare MemoryError", MemoryError(), True),
    ("a bare MemoryError with text", MemoryError("out of memory"), True),
    ("the CUDA OOM phrase (future GPU builds)",
     RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), True),
    ("a corrupt-file RuntimeError — must NOT be classified as OOM",
     RuntimeError("Error while deserializing header: HeaderTooLarge"), False),
    ("an unrelated RuntimeError — must NOT be classified as OOM",
     RuntimeError("size mismatch for model.encoder.weight"), False),
):
    check(label, accurate_engine._looks_like_out_of_memory(exc) is expected)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 4. load_engine() turns a real-shaped OOM failure into a clear, "
      "plain-language error — and leaves nothing half-loaded ===")
# ════════════════════════════════════════════════════════════════════════════
accurate_engine._engine_cache.clear()

real_from_pretrained_processor = None
real_from_pretrained_model = None


def install_failing_from_pretrained(exc_to_raise, fail_on="model"):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    global real_from_pretrained_processor, real_from_pretrained_model
    real_from_pretrained_processor = WhisperProcessor.from_pretrained
    real_from_pretrained_model = WhisperForConditionalGeneration.from_pretrained

    if fail_on == "model":
        WhisperProcessor.from_pretrained = classmethod(lambda cls, *a, **kw: real_from_pretrained_processor(*a, **kw))

        def failing_model_from_pretrained(*a, **kw):
            raise exc_to_raise
        WhisperForConditionalGeneration.from_pretrained = classmethod(
            lambda cls, *a, **kw: failing_model_from_pretrained(*a, **kw))
    else:
        def failing_processor_from_pretrained(*a, **kw):
            raise exc_to_raise
        WhisperProcessor.from_pretrained = classmethod(
            lambda cls, *a, **kw: failing_processor_from_pretrained(*a, **kw))


def restore_from_pretrained():
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    if real_from_pretrained_processor is not None:
        WhisperProcessor.from_pretrained = real_from_pretrained_processor
    if real_from_pretrained_model is not None:
        WhisperForConditionalGeneration.from_pretrained = real_from_pretrained_model


# Use a directory that passes check_model_available() so we actually reach
# the from_pretrained() calls the guard sits in front of. The real downloaded
# model directory does this without needing a throwaway fixture.
REAL_MODEL_DIR = os.path.expanduser("~/.stillscript_models/accurate-af-large-v3")
if not os.path.isdir(REAL_MODEL_DIR):
    print("  [SKIP] sections 4-6 need the real downloaded Accurate model "
        f"({REAL_MODEL_DIR}) — not faking a multi-GB directory just to pass.")
else:
    install_failing_from_pretrained(RuntimeError(REAL_CPU_OOM_TEXT), fail_on="model")
    try:
        accurate_engine.load_engine(REAL_MODEL_DIR)
        check("an OOM-shaped load failure raises", False, "no exception")
    except AccurateEngineUnavailable as e:
        check("an OOM-shaped load failure raises AccurateEngineUnavailable "
              "(not a bare RuntimeError reaching the UI)", True)
        msg = str(e)
        check("...with a plain-language message mentioning memory",
              "memory" in msg.lower(), msg)
        check("...that suggests a concrete fix (free memory / use Fast mode)",
              "fast mode" in msg.lower() and "close" in msg.lower(), msg)
        check("...and it is NOT the raw torch allocator text — a legal/clinical "
              "user should never see 'DefaultCPUAllocator'",
              "DefaultCPUAllocator" not in msg, msg)
    finally:
        restore_from_pretrained()
    check("nothing was cached from the failed attempt",
          REAL_MODEL_DIR not in accurate_engine._engine_cache)

    print("\n=== 5. A non-OOM failure is NOT relabelled as a memory problem ===")
    install_failing_from_pretrained(
        RuntimeError("Error while deserializing header: HeaderTooLarge"), fail_on="model")
    try:
        accurate_engine.load_engine(REAL_MODEL_DIR)
        check("a non-OOM load failure raises", False, "no exception")
    except AccurateEngineUnavailable as e:
        check("a non-OOM failure was wrongly turned into AccurateEngineUnavailable "
              "with the generic OOM message", False, str(e))
    except RuntimeError as e:
        check("a non-OOM RuntimeError propagates with its OWN message unchanged",
              "HeaderTooLarge" in str(e), str(e))
    finally:
        restore_from_pretrained()
    check("nothing was cached from this failed attempt either",
          REAL_MODEL_DIR not in accurate_engine._engine_cache)

    # ════════════════════════════════════════════════════════════════════════
    print("\n=== 6. Recovery: after a real, simulated load failure, Fast mode "
          "still works — a REAL Medium load, not assumed ===")
    # ════════════════════════════════════════════════════════════════════════
    ds._model_cache.clear()
    ds.get_model("medium")  # Fast mode "already in use" before the failed switch
    check("Fast mode is loaded going in", "medium" in ds._model_cache)

    install_failing_from_pretrained(RuntimeError(REAL_CPU_OOM_TEXT), fail_on="model")
    try:
        ds.transcribe_audio_accurate(CLIP, language="af", task="transcribe",
                                     model_dir=REAL_MODEL_DIR)
        check("the simulated Accurate-mode failure actually raised", False,
              "no exception — the failure injection did not work")
    except AccurateEngineUnavailable:
        check("Accurate mode failed as expected (simulated)", True)
    finally:
        restore_from_pretrained()

    check("Fast mode's cache was released before the failed attempt "
          "(release_fast_mode_model() still ran first, failure or not)",
          not ds._model_cache)

    # The actual recovery proof: really use Fast mode again, right now.
    result = ds.transcribe_audio(CLIP, language="af", task="transcribe", model_name="medium")
    check("Fast mode produces a real transcript after the failed Accurate-mode "
          "attempt in the same session",
          bool(result.get("text", "").strip()), result.get("text", "")[:100])
    check("Fast mode's cache is populated again (a clean reload, not a broken state)",
          "medium" in ds._model_cache)


# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
