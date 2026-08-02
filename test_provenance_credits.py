"""Tests for masterplan 2.6 (provenance extension) + 2.7 (credits).

WHAT IS AND IS NOT HERE
────────────────────────
Section 5 uses the REAL downloaded Accurate model directory
(~/.stillscript_models/accurate-af-large-v3) to run a REAL, cheap (probe-mode,
~1s) verify_merged_model() call and read its REAL completion stamp — so the
model-identity fields (repo_id / revision / layout) and the guard-mode label
are proven against real data, not fabricated. It does NOT load the ~6 GB
model into memory or run generate() — that's covered elsewhere
(test_accurate_engine.py, test_accurate_memory.py) and would make this file
far too slow to re-run routinely. If that directory is missing, section 5's
real-stamp checks are skipped (reported, not silently passed) and the
synthetic-guard-report checks still run.

Everything else here is pure/fast: dict and string-building logic only, no
I/O beyond one real small JSON read.

Run with the full-requirements venv (needs whisper + torch + transformers
importable, same as the other accurate-mode test files):
    <appvenv>/bin/python3 test_provenance_credits.py
"""

import os
import re
import sys
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stillscript as ds  # noqa: E402
import accurate_engine  # noqa: E402
import accurate_model_download as amd  # noqa: E402

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


REAL_MODEL_DIR = os.path.expanduser("~/.stillscript_models/accurate-af-large-v3")


# ════════════════════════════════════════════════════════════════════════════
print("=== 1. build_provenance()/format_provenance_lines() — Fast mode is "
      "byte-for-byte unchanged ===")
# ════════════════════════════════════════════════════════════════════════════
fast_prov = ds.build_provenance(
    mode="Fast (Vinnig)",
    language_label="Afrikaans",
    task="transcribe",
    diarized=True,
    num_speakers=3,
)

check("Fast-mode provenance dict has exactly the pre-2.6 keys, nothing added",
      set(fast_prov.keys()) == {"engine", "mode", "language", "task",
                                "diarized", "num_speakers", "timestamp"},
      sorted(fast_prov.keys()))
for new_key in ("model_id", "model_revision", "model_layout", "guard_verification"):
    check(f"...specifically, '{new_key}' is absent (not even None-valued) "
          "when the caller never passed it",
          new_key not in fast_prov)

expected_fast_lines = [
    f"Engine: {fast_prov['engine']}",
    "Mode: Fast (Vinnig)",
    "Language setting: Afrikaans",
    "Task: Original language",
    "Speaker identification: Yes (up to 3 speakers)",
    f"Generated: {fast_prov['timestamp']}",
]
actual_fast_lines = ds.format_provenance_lines(fast_prov)
check("Fast-mode footer is EXACTLY the same 6 lines, same order, as before "
      "this wave (no Accurate-only lines inserted)",
      actual_fast_lines == expected_fast_lines,
      actual_fast_lines)
check("Fast-mode engine label defaults to FAST_MODE_ENGINE_LABEL",
      fast_prov["engine"] == ds.FAST_MODE_ENGINE_LABEL)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 2. build_provenance() — new Accurate-mode fields are additive ===")
# ════════════════════════════════════════════════════════════════════════════
acc_prov = ds.build_provenance(
    mode="Accurate (Akkuraat)",
    language_label="Afrikaans",
    task="transcribe",
    diarized=False,
    engine=accurate_engine.ACCURATE_ENGINE_LABEL,
    model_id="DanieClar/stillscript-whisper-large-v3-afrikaans",
    model_revision="6baf2473d04da504f039ade149512d891e4a7ca5",
    model_layout="chunked",
    guard_verification="Full (1259 of 1259 tensors hashed)",
)
check("Accurate-mode provenance carries the engine label (not Fast's)",
      acc_prov["engine"] == accurate_engine.ACCURATE_ENGINE_LABEL)
check("...and all four new fields, unmodified",
      acc_prov["model_id"] == "DanieClar/stillscript-whisper-large-v3-afrikaans"
      and acc_prov["model_revision"] == "6baf2473d04da504f039ade149512d891e4a7ca5"
      and acc_prov["model_layout"] == "chunked"
      and acc_prov["guard_verification"] == "Full (1259 of 1259 tensors hashed)")

acc_lines = ds.format_provenance_lines(acc_prov)
check("Accurate-mode footer inserts the 4 new lines between 'Speaker "
      "identification' and 'Generated', existing lines otherwise untouched",
      acc_lines == [
          f"Engine: {accurate_engine.ACCURATE_ENGINE_LABEL}",
          "Mode: Accurate (Akkuraat)",
          "Language setting: Afrikaans",
          "Task: Original language",
          "Speaker identification: No",
          "Model: DanieClar/stillscript-whisper-large-v3-afrikaans",
          "Model revision: 6baf2473d04da504f039ade149512d891e4a7ca5",
          "Download layout: chunked",
          "Guard verification: Full (1259 of 1259 tensors hashed)",
          f"Generated: {acc_prov['timestamp']}",
      ],
      acc_lines)

# Partial case: an unmanaged model directory (no stamp) means
# model_revision/model_layout are None, but model_id and guard_verification
# are still known — the footer should show what IS known and quietly omit
# only the two genuinely-unknown lines, not fabricate them.
partial_prov = ds.build_provenance(
    mode="Accurate (Akkuraat)", language_label="Afrikaans", task="transcribe",
    diarized=False, engine=accurate_engine.ACCURATE_ENGINE_LABEL,
    model_id=amd.REPO_ID, model_revision=None, model_layout=None,
    guard_verification="Probe (9 sample tensors)",
)
partial_lines = ds.format_provenance_lines(partial_prov)
check("Unmanaged-directory case: 'Model' and 'Guard verification' lines "
      "still appear...",
      f"Model: {amd.REPO_ID}" in partial_lines
      and "Guard verification: Probe (9 sample tensors)" in partial_lines)
check("...but 'Model revision' and 'Download layout' are omitted rather "
      "than showing 'None'",
      not any(l.startswith("Model revision:") or l.startswith("Download layout:")
             for l in partial_lines),
      partial_lines)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 3. accurate_model_download.describe_verified_model() — stamp -> "
      "revision/layout mapping ===")
# ════════════════════════════════════════════════════════════════════════════
check("no stamp -> None (not an error)",
      amd.describe_verified_model(tempfile.mkdtemp()) is None)

with tempfile.TemporaryDirectory() as td:
    stamp = {
        "repo_id": amd.REPO_ID, "revision": amd.REVISION,
        "verified_at": "x", "full_verification": True,
        "guard": {}, "total_bytes": 123,
    }
    with open(os.path.join(td, amd.STAMP_FILENAME), "w") as fh:
        json.dump(stamp, fh)
    info = amd.describe_verified_model(td)
    check("single-file REVISION stamp -> layout='single'",
          info is not None and info["layout"] == "single", info)
    check("...revision/repo_id passed through unchanged",
          info["revision"] == amd.REVISION and info["repo_id"] == amd.REPO_ID)

with tempfile.TemporaryDirectory() as td:
    stamp = {
        "repo_id": amd.REPO_ID, "revision": amd.CHUNKED_REVISION,
        "verified_at": "x", "full_verification": True,
        "guard": {}, "total_bytes": 123,
    }
    with open(os.path.join(td, amd.STAMP_FILENAME), "w") as fh:
        json.dump(stamp, fh)
    info = amd.describe_verified_model(td)
    check("chunked CHUNKED_REVISION stamp -> layout='chunked'",
          info is not None and info["layout"] == "chunked", info)

with tempfile.TemporaryDirectory() as td:
    stamp = {
        "repo_id": amd.REPO_ID, "revision": "deadbeef" * 5,
        "verified_at": "x", "full_verification": True,
        "guard": {}, "total_bytes": 123,
    }
    with open(os.path.join(td, amd.STAMP_FILENAME), "w") as fh:
        json.dump(stamp, fh)
    info = amd.describe_verified_model(td)
    check("an unrecognised revision -> layout='unknown', not a guess",
          info is not None and info["layout"] == "unknown", info)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 4. accurate_engine._describe_model_provenance() — guard-mode "
      "label wording ===")
# ════════════════════════════════════════════════════════════════════════════
with tempfile.TemporaryDirectory() as td:
    # Unmanaged directory: no stamp.
    probe_report = {"ok": True, "model_dir": td, "probes_checked": 9,
                    "tensor_count": 1259, "full": False, "tensors_hashed": 9}
    info = accurate_engine._describe_model_provenance(td, probe_report)
    check("probe-mode guard_report -> 'Probe (N sample tensors)' wording",
          info["guard_verification"] == "Probe (9 sample tensors)", info)
    check("unmanaged (unstamped) dir -> model_id falls back to REPO_ID, "
          "revision/layout are None (not guessed)",
          info["model_id"] == amd.REPO_ID
          and info["model_revision"] is None
          and info["model_layout"] is None)

    full_report = {"ok": True, "model_dir": td, "probes_checked": 9,
                   "tensor_count": 1259, "full": True, "tensors_hashed": 1259}
    info2 = accurate_engine._describe_model_provenance(td, full_report)
    check("full-mode guard_report -> 'Full (N of M tensors hashed)' wording",
          info2["guard_verification"] == "Full (1259 of 1259 tensors hashed)", info2)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 5. End-to-end against the REAL managed model directory ===")
# ════════════════════════════════════════════════════════════════════════════
if not os.path.isdir(REAL_MODEL_DIR):
    print(f"  [SKIP] real model directory not found ({REAL_MODEL_DIR}) — "
         "not faking a multi-GB stamped directory just to pass.")
else:
    real_stamp_info = amd.describe_verified_model(REAL_MODEL_DIR)
    check("real managed directory has a real stamp", real_stamp_info is not None)
    check("...repo_id matches the pinned REPO_ID",
          real_stamp_info["repo_id"] == amd.REPO_ID, real_stamp_info)
    check("...revision is one of the two pinned, equivalent revisions",
          real_stamp_info["revision"] in amd.EQUIVALENT_REVISIONS, real_stamp_info)
    check("...layout is 'chunked' or 'single', never 'unknown', for a real "
          "download this app produced itself",
          real_stamp_info["layout"] in ("chunked", "single"), real_stamp_info)
    check("...the stamp itself records full verification (download-time "
          "guarantee, masterplan 2.1a.2/2.1a.3, unconditional)",
          real_stamp_info["full_verification"] is True, real_stamp_info)

    # A REAL, cheap (~1s, probe-mode default) guard check on the real model,
    # exactly what load_engine() itself would compute on a cache miss.
    real_guard_report = accurate_engine.verify_merged_model(REAL_MODEL_DIR)
    real_info = accurate_engine._describe_model_provenance(REAL_MODEL_DIR, real_guard_report)
    check("real end-to-end: model_id/revision/layout come from the real stamp",
          real_info["model_id"] == amd.REPO_ID
          and real_info["model_revision"] == real_stamp_info["revision"]
          and real_info["model_layout"] == real_stamp_info["layout"],
          real_info)
    check("real end-to-end: guard_verification reflects the mode that ACTUALLY "
          "ran just now (default is probe, per masterplan 2.2a)",
          real_info["guard_verification"].startswith("Probe")
          if not real_guard_report.get("full")
          else real_info["guard_verification"].startswith("Full"),
          real_info["guard_verification"])


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 6. CREDITS (masterplan 2.7) — André Oosthuizen's model + "
      "dataset, CC-BY-4.0 ===")
# ════════════════════════════════════════════════════════════════════════════
check("Whisper's existing entry is untouched (name/license/url unchanged)",
      ds.CREDITS[0] == {
          "name": "OpenAI Whisper",
          "detail": ('Radford et al., "Robust Speech Recognition via '
                     'Large-Scale Weak Supervision", arXiv:2212.04356'),
          "license": "Apache License 2.0",
          "url": "https://arxiv.org/abs/2212.04356",
      })

adapter_entries = [c for c in ds.CREDITS if "afrikaans" in c["url"].lower()
                   and "datasets/" not in c["url"]]
dataset_entries = [c for c in ds.CREDITS if "datasets/" in c["url"]]

check("exactly one adapter-model credit entry added", len(adapter_entries) == 1,
      adapter_entries)
check("exactly one dataset credit entry added", len(dataset_entries) == 1,
      dataset_entries)

if adapter_entries:
    adapter = adapter_entries[0]
    check("adapter entry license is CC-BY-4.0", adapter["license"] == "CC-BY-4.0")
    check("adapter entry URL is the exact HF repo from the fetched model card",
          adapter["url"] == "https://huggingface.co/andreoosthuizen/whisper-large-v3-afrikaans")
    check("adapter entry credits the creator by name (CC-BY-4.0 'Author')",
          "Oosthuizen" in adapter["detail"])
    check("adapter entry names the work being credited (CC-BY-4.0 'Title')",
          "Whisper Large V3 Afrikaans" in adapter["name"])
    check("adapter entry indicates modifications were made (CC-BY-4.0 "
          "requirement — merging counts as a modification of the adapter's "
          "original standalone form)",
          "merg" in adapter["detail"].lower())

if dataset_entries:
    dataset = dataset_entries[0]
    check("dataset entry license is CC-BY-4.0", dataset["license"] == "CC-BY-4.0")
    check("dataset entry URL is the exact HF dataset repo from the fetched "
          "model card's hyperlink (not a guessed path)",
          dataset["url"] == "https://huggingface.co/datasets/andreoosthuizen/afrikaans-30s")
    check("dataset entry credits Oosthuizen by name",
          "Oosthuizen" in dataset["detail"])

check("credit appears regardless of which HF layout/revision was actually "
      "downloaded — CREDITS is a static module-level list, not conditioned "
      "on any revision/layout value",
      "model_revision" not in str(ds.CREDITS) and "model_layout" not in str(ds.CREDITS))


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 7. Provenance stays export-layer-only — current_transcript is "
      "never touched by it (structural check on the actual source) ===")
# ════════════════════════════════════════════════════════════════════════════
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stillscript.py"),
          encoding="utf-8").read()

assignments = re.findall(r"self\.current_transcript\s*=\s*(.+)", src)
check("current_transcript is assigned from plain transcript text only — "
      "never from provenance/build_provenance/format_provenance_lines",
      len(assignments) >= 2 and all(
          "provenance" not in rhs.lower() for rhs in assignments
      ),
      assignments)

format_call_lines = [
    line for line in src.splitlines()
    if "format_provenance_lines(provenance)" in line
    and not line.strip().startswith(("def ", "#"))
]
check("format_provenance_lines() is called at exactly two real call sites in "
      "the whole file (the .txt footer in _save_transcript, the .docx footer "
      "in _make_docx) — i.e. still export-layer-only, not a third call site "
      "that could leak into the summary prompt",
      len(format_call_lines) == 2, format_call_lines)


# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
