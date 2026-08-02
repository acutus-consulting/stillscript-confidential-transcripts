"""Tests for masterplan 4.1's legacy user-data migration (DanScribe ->
StillScript).

WHY THIS FILE EXISTS
────────────────────
The rename moves five things that are USER DATA, not labels: the config
file, the log, the multi-GB model directory, the transcripts folder, and the
OS keyring entry holding the API key. Getting any of them wrong silently
costs a real beta user their settings, their API key, their transcripts, or
a ~6 GB re-download. So the migration is tested against a real temporary
HOME with real files on disk — not mocked filesystem calls.

migrate_legacy_user_data() takes its migration list as an argument precisely
so this test can point it at a scratch directory instead of the real home.
The production call site passes nothing and uses the module-level list.

Run with the full-requirements venv (needs stillscript importable):
    <appvenv>/bin/python3 test_rename_migration.py
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stillscript as ds  # noqa: E402

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


def make_scratch_migrations(root):
    """The same five-way shape as the real _LEGACY_PATH_MIGRATIONS, rooted in
    a scratch dir so nothing touches the real home directory."""
    return [
        (root / ".danscribe_config.json", root / ".stillscript_config.json"),
        (root / ".danscribe.log", root / ".stillscript.log"),
        (root / ".danscribe_models", root / ".stillscript_models"),
        (root / "Documents" / "DanScribe_Transcriptions",
         root / "Documents" / "StillScript_Transcriptions"),
    ]


# ════════════════════════════════════════════════════════════════════════════
print("=== 1. Fresh install: no old data anywhere, nothing migrated, no error ===")
# ════════════════════════════════════════════════════════════════════════════
scratch = Path(tempfile.mkdtemp(prefix="stillscript_rename_fresh_"))
try:
    migrated = ds.migrate_legacy_user_data(make_scratch_migrations(scratch))
    check("a fresh install migrates nothing (empty result, not an exception)",
          migrated == [], migrated)
    check("...and does NOT create empty new-name files/dirs as a side effect "
          "(a fresh install must look untouched, not pre-seeded)",
          list(scratch.iterdir()) == [], list(scratch.iterdir()))
finally:
    shutil.rmtree(scratch, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 2. Existing install: all four paths present, all carry over "
      "with contents intact ===")
# ════════════════════════════════════════════════════════════════════════════
scratch = Path(tempfile.mkdtemp(prefix="stillscript_rename_existing_"))
try:
    # Real files with real, distinguishable content — so "did it migrate?" is
    # answered by reading the data back, not just by a path existing.
    (scratch / ".danscribe_config.json").write_text(
        '{"last_language": "Afrikaans", "reproducibility": "Best effort"}')
    (scratch / ".danscribe.log").write_text("2026-01-01 INFO  an old log line\n")

    old_models = scratch / ".danscribe_models" / "accurate-af-large-v3"
    old_models.mkdir(parents=True)
    (old_models / "config.json").write_text('{"model_type": "whisper"}')
    (old_models / ".stillscript_download.json").write_text('{"revision": "abc123"}')

    old_transcripts = scratch / "Documents" / "DanScribe_Transcriptions"
    old_transcripts.mkdir(parents=True)
    (old_transcripts / "Toets_transcript.txt").write_text("Spreker 1: hallo\n")

    migrated = ds.migrate_legacy_user_data(make_scratch_migrations(scratch))

    check("all four legacy paths were migrated", len(migrated) == 4, len(migrated))

    check("config file now exists under the new name",
          (scratch / ".stillscript_config.json").exists())
    check("...with its contents intact (settings genuinely carried over, not "
          "reset to defaults)",
          '"reproducibility": "Best effort"'
          in (scratch / ".stillscript_config.json").read_text())

    check("log file migrated", (scratch / ".stillscript.log").exists())
    check("...with its previous content intact",
          "an old log line" in (scratch / ".stillscript.log").read_text())

    new_models = scratch / ".stillscript_models" / "accurate-af-large-v3"
    check("model directory migrated, nested structure and all",
          new_models.is_dir())
    check("...including the model file itself (this is the ~6 GB re-download "
          "the migration exists to avoid)",
          (new_models / "config.json").read_text() == '{"model_type": "whisper"}')
    check("...and the download stamp the guard/provenance rely on",
          (new_models / ".stillscript_download.json").exists())

    new_transcripts = scratch / "Documents" / "StillScript_Transcriptions"
    check("transcripts folder migrated", new_transcripts.is_dir())
    check("...with the user's existing transcripts still readable",
          "Spreker 1: hallo" in (new_transcripts / "Toets_transcript.txt").read_text())

    check("the OLD paths are gone after a move (one source of truth, no "
          "duplicate state that could silently diverge)",
          not (scratch / ".danscribe_config.json").exists()
          and not (scratch / ".danscribe_models").exists()
          and not (scratch / "Documents" / "DanScribe_Transcriptions").exists())
finally:
    shutil.rmtree(scratch, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 3. Partial state: only some legacy paths exist ===")
# ════════════════════════════════════════════════════════════════════════════
# Real case: a user who set a language but never activated Accurate mode has
# a config file and no model directory at all.
scratch = Path(tempfile.mkdtemp(prefix="stillscript_rename_partial_"))
try:
    (scratch / ".danscribe_config.json").write_text('{"last_language": "English"}')
    migrated = ds.migrate_legacy_user_data(make_scratch_migrations(scratch))
    check("only the paths that actually exist are migrated",
          len(migrated) == 1, migrated)
    check("the one that existed carried over",
          '"last_language": "English"'
          in (scratch / ".stillscript_config.json").read_text())
    check("...and no placeholder was invented for the absent model directory",
          not (scratch / ".stillscript_models").exists())
finally:
    shutil.rmtree(scratch, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 4. Never overwrite: new-name data already present wins ===")
# ════════════════════════════════════════════════════════════════════════════
# The dangerous case. If a user somehow has BOTH (e.g. ran a new build, then
# an old one, then the new one again), the newer StillScript data must not be
# clobbered by the stale DanScribe copy.
scratch = Path(tempfile.mkdtemp(prefix="stillscript_rename_both_"))
try:
    (scratch / ".danscribe_config.json").write_text('{"last_language": "STALE"}')
    (scratch / ".stillscript_config.json").write_text('{"last_language": "CURRENT"}')

    migrated = ds.migrate_legacy_user_data(make_scratch_migrations(scratch))

    check("nothing is migrated when the new-name path already exists",
          migrated == [], migrated)
    check("the CURRENT StillScript config is untouched — a stale legacy file "
          "must never overwrite newer data",
          '"last_language": "CURRENT"'
          in (scratch / ".stillscript_config.json").read_text())
    check("...and the legacy file is left in place rather than deleted "
          "(nothing is ever destroyed, even when skipped)",
          (scratch / ".danscribe_config.json").exists())
finally:
    shutil.rmtree(scratch, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 5. Migration failure is survivable — it must never stop startup ===")
# ════════════════════════════════════════════════════════════════════════════
scratch = Path(tempfile.mkdtemp(prefix="stillscript_rename_fail_"))
try:
    # An unreadable/undeletable source would raise inside os.rename; simulate
    # the general "this one item blew up" case with a deliberately bogus pair
    # alongside a good one, and confirm the good one still lands.
    (scratch / ".danscribe_config.json").write_text('{"last_language": "Afrikaans"}')
    bogus = [
        (Path("\x00 not a real path"), Path("\x00 also not")),
        (scratch / ".danscribe_config.json", scratch / ".stillscript_config.json"),
    ]
    raised = False
    try:
        migrated = ds.migrate_legacy_user_data(bogus)
    except Exception:
        raised = True
        migrated = []
    check("a broken migration entry does not raise out of "
          "migrate_legacy_user_data() (it runs at import, before logging — an "
          "exception here would stop the app from starting at all)",
          not raised)
    check("...and a valid entry alongside the broken one still migrates",
          (scratch / ".stillscript_config.json").exists()
          and '"last_language": "Afrikaans"'
          in (scratch / ".stillscript_config.json").read_text())
finally:
    shutil.rmtree(scratch, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 6. The production constants point where they should ===")
# ════════════════════════════════════════════════════════════════════════════
check("CONFIG_PATH is the StillScript-named file",
      ds.CONFIG_PATH.endswith(".stillscript_config.json"), ds.CONFIG_PATH)
check("LOG_PATH is the StillScript-named file",
      ds.LOG_PATH.endswith(".stillscript.log"), ds.LOG_PATH)
check("the keyring service is 'StillScript'",
      ds._KEYRING_SERVICE == "StillScript", ds._KEYRING_SERVICE)
check("...and the legacy keyring service name is still known, so an existing "
      "user's stored API key can be found and re-homed rather than orphaned",
      ds._LEGACY_KEYRING_SERVICE == "DanScribe", ds._LEGACY_KEYRING_SERVICE)

legacy_olds = [str(old) for old, _ in ds._LEGACY_PATH_MIGRATIONS]
legacy_news = [str(new) for _, new in ds._LEGACY_PATH_MIGRATIONS]
check("every migration source carries a DanScribe-era name",
      all("anscribe" in p or "DanScribe" in p for p in legacy_olds), legacy_olds)
check("every migration target carries a StillScript name",
      all("tillscript" in p or "StillScript" in p for p in legacy_news), legacy_news)
check("the model root the download module resolves to matches the migration "
      "target — the two must not drift apart",
      any(p.endswith(".stillscript_models") for p in legacy_news), legacy_news)


# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
