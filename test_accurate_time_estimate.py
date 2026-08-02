"""Tests for masterplan 2.4 (Accurate-mode time estimate / progress UI)
that don't need a real Tkinter app or a real generate() call.

WHAT IS AND IS NOT HERE
────────────────────────
Pure/fast logic only: _estimate_accurate_time_range()'s arithmetic across a
few different file durations and both Reproducibility settings, and the
ACCURATE_*_MULTIPLIER_RANGE constants' own shape/sanity. No torch, no real
model load, no real audio.

The REAL UI wiring — that AccurateTimeEstimateDialog actually appears with
the right numbers before a run starts, that the real progress_callback
wired into _transcribe_with_accurate_model() actually reaches the real
progress_bar/status_label widgets, and that the UI stays correct if that
callback fires zero times or is handed something unexpected — is proven
separately in test_accurate_ui.py (section 6), which already has the
real-Tkinter-app-under-Xvfb harness this would otherwise have to duplicate.

Run with the full-requirements venv (needs accurate_engine importable, same
as the other accurate-mode test files):
    <appvenv>/bin/python3 test_accurate_time_estimate.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import DanScribe_v2 as ds  # noqa: E402
import accurate_engine  # noqa: E402

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


# ════════════════════════════════════════════════════════════════════════════
print("=== 1. The multiplier constants themselves are sane ===")
# ════════════════════════════════════════════════════════════════════════════
for choice in ("Consistent", "Best effort"):
    check(f"'{choice}' has an entry in ACCURATE_TRANSCRIPTION_MULTIPLIER_RANGE",
          choice in ds.ACCURATE_TRANSCRIPTION_MULTIPLIER_RANGE)
    low, high = ds.ACCURATE_TRANSCRIPTION_MULTIPLIER_RANGE[choice]
    check(f"'{choice}': low <= high ({low} <= {high})", low <= high)
    check(f"'{choice}': both bounds are positive", low > 0 and high > 0)

c_low, c_high = ds.ACCURATE_TRANSCRIPTION_MULTIPLIER_RANGE["Consistent"]
b_low, b_high = ds.ACCURATE_TRANSCRIPTION_MULTIPLIER_RANGE["Best effort"]
check("Consistent's range is narrower than Best effort's (golf 2.8/2.10's "
      "actual finding — Best effort's retries are what cause its wide "
      "spread on hard audio, not Consistent's)",
      (c_high - c_low) < (b_high - b_low),
      f"Consistent spread={c_high - c_low:.2f}, Best effort spread={b_high - b_low:.2f}")

d_low, d_high = ds.ACCURATE_DIARIZATION_MULTIPLIER_RANGE
check("diarization multiplier range: low <= high", d_low <= d_high)
check("diarization multiplier range: both bounds positive", d_low > 0 and d_high > 0)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 2. _estimate_accurate_time_range() across several durations, "
      "both settings, diarization on/off ===")
# ════════════════════════════════════════════════════════════════════════════
# Real durations this product actually sees: a short clip, golf 2.10's own
# 25-min representative clip, and an hour-plus file (the diarization long-
# file path's own territory, even though that gate itself is dead code).
DURATIONS_SECONDS = [30.0, 1426.9, 3600.0]

for duration in DURATIONS_SECONDS:
    for choice in ("Consistent", "Best effort"):
        for diarize in (False, True):
            result = ds._estimate_accurate_time_range(duration, choice, diarize)
            label = f"duration={duration}s, repro={choice!r}, diarize={diarize}"

            check(f"{label}: returns a (low, high) pair, not None, for a known duration",
                  result is not None and len(result) == 2, result)
            if result is None:
                continue
            low, high = result

            check(f"{label}: low <= high", low <= high, result)
            check(f"{label}: low is at least the duration itself (no run is "
                  f"faster than real time)", low >= duration, result)

            t_low, t_high = ds.ACCURATE_TRANSCRIPTION_MULTIPLIER_RANGE[choice]
            expected_low = duration * t_low
            expected_high = duration * t_high
            if diarize:
                expected_low += duration * d_low
                expected_high += duration * d_high
            check(f"{label}: matches the multiplier arithmetic exactly, not "
                  f"just plausibly",
                  abs(low - expected_low) < 1e-9 and abs(high - expected_high) < 1e-9,
                  f"got=({low}, {high}), expected=({expected_low}, {expected_high})")

            if diarize:
                no_diarize_low, no_diarize_high = ds._estimate_accurate_time_range(
                    duration, choice, False
                )
                check(f"{label}: enabling diarization strictly increases both "
                      f"bounds versus the same duration/setting without it",
                      low > no_diarize_low and high > no_diarize_high,
                      f"with={result}, without=({no_diarize_low}, {no_diarize_high})")


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 3. Unknown/missing duration and unrecognised setting — the two "
      "'this should never happen but must not crash' inputs ===")
# ════════════════════════════════════════════════════════════════════════════
check("duration_seconds=None returns None (mirrors "
      "_probe_audio_duration_seconds()'s own 'unknown' contract), not a crash",
      ds._estimate_accurate_time_range(None, "Consistent", False) is None)

check("an unrecognised reproducibility string falls back to Consistent's "
      "range rather than raising (same fail-safe convention load_config() "
      "already uses for this exact config key)",
      ds._estimate_accurate_time_range(100.0, "not a real choice", False)
      == ds._estimate_accurate_time_range(100.0, "Consistent", False))


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 4. Consistent's range genuinely reflects the real golf 2.8/2.10 "
      "measurements, not placeholder numbers ===")
# ════════════════════════════════════════════════════════════════════════════
# These are the exact figures from the delivered results, not re-derived —
# a regression here means the UI copy has drifted from what was actually
# measured.
check("Consistent low bound = 2.98x (golf 2.10, clean 25-min representative clip)",
      c_low == 2.98, c_low)
check("Consistent high bound = 3.08x (golf 2.8, extreme-noise bar clip)",
      c_high == 3.08, c_high)
check("Best effort low bound = 3.06x (golf 2.8/2.10's clean-audio baseline)",
      b_low == 3.06, b_low)
check("Best effort high bound = 9.46x (golf 2.8's extreme-noise bar clip)",
      b_high == 9.46, b_high)

# Golf 2.8's mixed-language measurement (7.175x) must fall inside Best
# effort's stated range — the whole point of that range is to honestly
# cover every measured Best-effort data point, not just the two extremes.
MIXED_LANGUAGE_MULTIPLIER = 7.175
check("the mixed-language 46-min clip's measured 7.175x (Best effort) falls "
      "inside the stated Best-effort range — the range genuinely covers all "
      "three measured conditions, not just two of three",
      b_low <= MIXED_LANGUAGE_MULTIPLIER <= b_high, (b_low, MIXED_LANGUAGE_MULTIPLIER, b_high))


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 5. accurate_engine.PROGRESS_FRAMES_PER_SECOND — the conversion "
      "factor the real progress UI depends on ===")
# ════════════════════════════════════════════════════════════════════════════
check("PROGRESS_FRAMES_PER_SECOND is 100.0 (WhisperFeatureExtractor's fixed "
      "hop length — verified empirically against a real processor call, "
      "see accurate_engine.py's own comment)",
      accurate_engine.PROGRESS_FRAMES_PER_SECOND == 100.0,
      accurate_engine.PROGRESS_FRAMES_PER_SECOND)


# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
