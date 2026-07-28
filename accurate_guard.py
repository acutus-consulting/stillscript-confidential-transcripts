"""Startup guard for the Accurate-mode model (masterplan 2.2).

WHY THIS EXISTS
───────────────
Accurate mode's whole value proposition is that the transcript can be trusted
for legal and clinical work. The failure mode that destroys that is not a
crash — it is loading the *wrong* model and producing fluent, plausible,
subtly-wrong Afrikaans. Stock Whisper large-v3 will happily transcribe
Afrikaans and produce output that looks fine to anyone not checking closely.
So a silent fall back to a base or corrupted checkpoint is worse than a loud
refusal to run, and this module exists to make that refusal loud.

WHAT IT ACTUALLY CHECKS
───────────────────────
The masterplan phrased this as a "non-zero tensor count" check. That check was
right where it originally lived — inside verify_and_merge.py, applied to the
LoRA *adapter file* before merging, where an empty adapter really is the
failure. Applied to the *merged* model it is nearly useless: stock large-v3
also has 1259 non-zero tensors, so a tensor-count guard passes on precisely
the model we are trying to catch. This module therefore checks something
stronger — that specific weights carry the fine-tuning:

  1. STRUCTURE   — expected tensor count, every probe present, right shape and
                   dtype, no NaN/Inf. Catches truncated or partial downloads.

  2. FINE-TUNE EVIDENCE (the anti-base check) — for six decoder attention
     tensors, the values must differ from stock large-v3's values by a
     non-trivial amount. The pinned fingerprint records what base large-v3
     holds at those positions, so when the check fails the guard can say
     "this is stock Whisper" rather than just "mismatch".

  3. LINEAGE     — three tensors that a q_proj/v_proj LoRA can never touch
     (conv1, a k_proj, a layer norm) must still match base large-v3 exactly.
     Together with (2) this is two-sided: the model must be large-v3, *and*
     must have been adapted in the decoder. A different model that happens to
     differ from base fails (3); a stock model fails (2).

  4. IDENTITY    — sha256 of each probe tensor against the approved merge.
     Catches a re-quantised, re-merged, or otherwise different-but-plausible
     model, and any corruption inside the probed tensors.

FULL VERIFICATION (opt-in)
──────────────────────────
The nine probes are a sample, so corruption confined to an unprobed tensor
passes. Full mode hashes every tensor in the file against a pinned manifest,
which leaves nowhere for a bad byte to hide.

Measured on Danie's machine against the real 5.75 GiB merge (median of 3):

                    cold page cache     warm
    default (9 probes)     1.15 s       0.05 s
    full (1259 tensors)    5.25 s       3.70 s

So full verification costs about four extra seconds — cheap next to a run that
takes ~3x real time. It is still opt-in rather than default because it is pure
insurance: the probes already catch a wrong or base model, and full mode only
adds coverage of corruption hiding in an unprobed tensor. Worth turning on
before a large client batch, or whenever you have reason to suspect the file
(an interrupted copy, a flaky disk, a model from somewhere you did not
control). Given how small the cost turned out to be, making it the default is
a reasonable call if you would rather not have to remember.

Enable it the same way the model directory is chosen — explicit argument wins,
then the environment variable:

    verify_merged_model(model_dir, full=True)
    STILLSCRIPT_ACCURATE_FULL_VERIFY=1

Note the asymmetry with NO BYPASS below: the environment may make this guard
STRICTER, never weaker. That is the only direction a runtime switch is allowed
to move.

NO BYPASS
─────────
There is deliberately no environment variable, flag, or argument that weakens
any of these checks. A guard that can be switched off at runtime is one
mis-set variable away from shipping a transcript from the wrong model, and the
whole point here is that the wrong model must stop the run. The supported path
for a deliberate re-merge is to re-run pin_accurate_fingerprint.py, which
re-pins the fingerprint to the new model after you have checked it — an
explicit, reviewable change to a committed file, not a runtime toggle.

GROUND TRUTH THIS IS BUILT ON (verified against the real files, 2026-07-27)
──────────────────────────────────────────────────────────────────────────
André's adapter is LoRA r=32, alpha=64, target_modules=[q_proj, v_proj].
It contains LoRA tensors for the encoder as well as the decoder — but every
one of the 64 encoder `lora_B` tensors is still EXACTLY ZERO, i.e. never
trained. Merging therefore leaves all 64 encoder q/v_proj weights bit-identical
to base large-v3; only the 128 decoder q/v_proj tensors actually move.

That is why every "finetuned" probe is a decoder tensor. A guard that sampled
an encoder q_proj would find it identical to base and permanently report a
stock model — a false alarm that would block the product. Measured sampled
relative deltas on the approved merge run 5.8e-2 to 2.9e-1, so the 1e-3
threshold below sits three orders of magnitude below the real signal and four
above float32 round-trip noise.

DEPENDENCIES
────────────
Deliberately light: `safetensors` + `torch`, imported lazily inside the
function. This module can be imported (for its exception types) without
dragging `transformers` into Fast mode's runtime.
"""

import base64
import hashlib
import json
import logging
import os

try:
    from accurate_fingerprint import ACCURATE_FINGERPRINT
except ImportError:  # pragma: no cover - only if the generated file is missing
    # Missing pinned data must fail loudly at verify time, never silently
    # degrade into "no checks performed".
    ACCURATE_FINGERPRINT = None

logger = logging.getLogger("danscribe.accurate.guard")

# Minimum sampled relative delta from base large-v3 for a "finetuned" probe.
# See the note above on why 1e-3 is the right order of magnitude.
MIN_RELATIVE_DELTA = 1e-3

# Opt-in full verification. Same resolution order as the model directory:
# explicit argument wins, then this variable. Truthy values only; anything
# else leaves the default (probes only) in place.
FULL_VERIFY_ENV_VAR = "STILLSCRIPT_ACCURATE_FULL_VERIFY"
_TRUTHY = ("1", "true", "yes", "on")

# Cap on how many mismatches a full-mode failure lists. A wholesale wrong file
# would otherwise produce a 1259-line exception message.
_MAX_REPORTED_FAILURES = 12


class AccurateEngineUnavailable(RuntimeError):
    """Raised when the Accurate engine cannot run.

    Deliberately a distinct exception type: the caller must be able to tell
    "Accurate mode is not available on this machine" apart from "transcription
    failed", and must never silently fall back to another engine.

    Defined here rather than in accurate_engine so that the UI can import and
    handle Accurate-mode errors without importing torch/transformers.
    """


class AccurateModelGuardError(AccurateEngineUnavailable):
    """Raised when the model on disk is not the approved Afrikaans merge.

    A subclass of AccurateEngineUnavailable so existing `except
    AccurateEngineUnavailable` handlers still catch it, while code that wants
    to distinguish "wrong model" from "missing dependency" can.

    `failures` holds one string per failed check, for logging.
    """

    def __init__(self, message, failures=None, model_dir=None):
        super().__init__(message)
        self.failures = failures or []
        self.model_dir = model_dir


def _weight_files(model_dir):
    """Return the safetensors files holding the weights, sharded or not."""
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as fh:
            shards = sorted(set(json.load(fh).get("weight_map", {}).values()))
        return [os.path.join(model_dir, s) for s in shards]
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(single):
        return [single]
    return []


def _sample_indices(numel, n):
    n = min(n, numel)
    step = numel / n
    return [min(numel - 1, int(i * step)) for i in range(n)]


def full_tensor_digest(key, tensor):
    """Content digest of one tensor, for the full-manifest check.

    Binds the key, dtype and shape into the hash alongside the raw bytes, so a
    tensor that was renamed, re-typed, or reshaped to the same byte length
    still fails. Hashes the bytes as stored (no fp32 cast) — full mode is about
    detecting change, and casting 6.2 GB would only make it slower.

    Shared with pin_accurate_fingerprint.py so the pinning and checking sides
    can never drift apart. Changing this invalidates every pinned full digest.
    """
    h = hashlib.sha256()
    h.update(f"{key}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
    # memoryview over the numpy view: hashes in place, no multi-hundred-MB copy.
    h.update(memoryview(tensor.contiguous().flatten().numpy()).cast("B"))
    return h.hexdigest()


def full_verify_requested(full=None):
    """Resolve whether to run full verification.

    Mirrors resolve_model_dir(): explicit argument wins, then the environment.
    """
    if full is not None:
        return bool(full)
    return (os.environ.get(FULL_VERIFY_ENV_VAR) or "").strip().lower() in _TRUTHY


def verify_merged_model(model_dir, fingerprint=None, full=None):
    """Verify `model_dir` holds the approved fine-tuned Afrikaans merge.

    Returns a report dict on success. Raises AccurateModelGuardError listing
    every failed check otherwise — all checks are run before raising, so one
    log line tells you the whole story rather than only the first problem.

    Every check is mandatory; there is no way to relax any of them at runtime.
    See NO BYPASS in the module docstring.

    By default reads only the probe tensors straight out of the safetensors
    file (~40 MB; 1.15 s cold, 0.05 s warm), not the whole 5.75 GiB model, and
    is intended to run BEFORE from_pretrained() so a bad model fails in about a
    second instead of after a full load.

    `full=True` (or STILLSCRIPT_ACCURATE_FULL_VERIFY=1) additionally hashes
    every tensor against the pinned manifest: 5.25 s cold, 3.70 s warm — about
    four seconds more than the default. Purely additive; all the default checks
    still run, first and unchanged.
    """
    import torch
    from safetensors import safe_open

    if fingerprint is None:
        fingerprint = ACCURATE_FINGERPRINT
    if fingerprint is None:
        raise AccurateModelGuardError(
            "No pinned fingerprint available (accurate_fingerprint.py is "
            "missing or unreadable), so the model cannot be verified and "
            "Accurate mode will not run. Regenerate it with "
            "pin_accurate_fingerprint.py.",
            failures=["no pinned fingerprint"], model_dir=model_dir,
        )

    do_full = full_verify_requested(full)
    failures = []

    files = _weight_files(model_dir)
    if not files:
        raise AccurateModelGuardError(
            f"No safetensors weights found in {model_dir}. Accurate mode cannot "
            f"verify the model, so it will not run.",
            failures=["no weight files"], model_dir=model_dir,
        )

    # Map every key to the shard holding it, and open each shard once.
    handles = {p: safe_open(p, framework="pt") for p in files}
    key_to_file = {}
    for path, fh in handles.items():
        for k in fh.keys():
            key_to_file[k] = path

    expected_count = fingerprint.get("tensor_count")
    if expected_count is not None and len(key_to_file) != expected_count:
        failures.append(
            f"tensor count is {len(key_to_file)}, expected {expected_count} "
            f"— the model file is not the expected large-v3 layout"
        )

    saw_base_model = False
    for probe in fingerprint["probes"]:
        key, role = probe["key"], probe["role"]
        if key not in key_to_file:
            failures.append(f"{key}: missing from the model")
            continue

        tensor = handles[key_to_file[key]].get_tensor(key)

        if list(tensor.shape) != list(probe["shape"]):
            failures.append(
                f"{key}: shape {list(tensor.shape)}, expected {probe['shape']}")
            continue
        # A dtype mismatch is a failure, but deliberately NOT a `continue`:
        # stock openai/whisper-large-v3 ships fp16, so the most likely
        # real-world wrong model is both re-quantised AND stock. If we skipped
        # the value checks here we would report the dtype and never notice it
        # was base Whisper — the generic message instead of the useful one.
        # The comparisons below run in fp32, so they work either way.
        if str(tensor.dtype).replace("torch.", "") != probe["dtype"]:
            failures.append(
                f"{key}: dtype {tensor.dtype}, expected {probe['dtype']} "
                f"— a re-quantised model is not the validated one")

        flat = tensor.reshape(-1).to(dtype=torch.float32)
        if not torch.isfinite(flat).all():
            failures.append(f"{key}: contains NaN/Inf — the file is corrupt")
            continue

        base_sample = torch.frombuffer(
            bytearray(base64.b64decode(probe["base_sample_b64"])), dtype=torch.float32)
        sample = flat[_sample_indices(flat.numel(), base_sample.numel())]
        rel = float((sample - base_sample).norm() / (base_sample.norm() + 1e-12))

        if role == "finetuned":
            if rel == 0.0:
                saw_base_model = True
                failures.append(
                    f"{key}: identical to stock Whisper large-v3 — the "
                    f"Afrikaans fine-tuning is NOT present")
            elif rel < MIN_RELATIVE_DELTA:
                failures.append(
                    f"{key}: differs from stock large-v3 by only {rel:.2e} "
                    f"(need >= {MIN_RELATIVE_DELTA:.0e}) — effectively un-fine-tuned")
        elif role == "lineage":
            if rel != 0.0:
                failures.append(
                    f"{key}: differs from stock large-v3 by {rel:.2e}, but a "
                    f"q_proj/v_proj LoRA cannot touch this tensor — this is not "
                    f"a large-v3 + Afrikaans-adapter merge")

        digest = hashlib.sha256(flat.numpy().tobytes()).hexdigest()
        if digest != probe["sha256"]:
            failures.append(
                f"{key}: sha256 {digest[:16]}… does not match the approved "
                f"merge ({probe['sha256'][:16]}…)")

    # ── Opt-in full pass ────────────────────────────────────────────────────
    # Everything above has already run; this only adds checks. Hashes every
    # tensor in the file against the pinned manifest, so corruption in a
    # tensor the probes do not cover has nowhere to hide.
    tensors_hashed = 0
    if do_full:
        manifest = fingerprint.get("full_manifest")
        if not manifest:
            raise AccurateModelGuardError(
                "Full verification was requested, but the pinned fingerprint "
                "has no full_manifest. Regenerate it with "
                "pin_accurate_fingerprint.py, or run without "
                f"{FULL_VERIFY_ENV_VAR}.",
                failures=["no full_manifest pinned"], model_dir=model_dir,
            )

        missing = sorted(set(manifest) - set(key_to_file))
        unexpected = sorted(set(key_to_file) - set(manifest))
        for key in missing[:_MAX_REPORTED_FAILURES]:
            failures.append(f"{key}: in the approved manifest but missing here")
        for key in unexpected[:_MAX_REPORTED_FAILURES]:
            failures.append(f"{key}: present here but not in the approved manifest")

        mismatched = []
        for key in sorted(set(manifest) & set(key_to_file)):
            digest = full_tensor_digest(key, handles[key_to_file[key]].get_tensor(key))
            tensors_hashed += 1
            if digest != manifest[key]:
                mismatched.append(key)
        for key in mismatched[:_MAX_REPORTED_FAILURES]:
            failures.append(
                f"{key}: content differs from the approved merge (full check)")
        if len(mismatched) > _MAX_REPORTED_FAILURES:
            failures.append(
                f"…and {len(mismatched) - _MAX_REPORTED_FAILURES} further "
                f"tensors differ ({len(mismatched)} of {len(manifest)} total)")

    if failures:
        if saw_base_model:
            headline = (
                "ACCURATE MODE REFUSED TO START: the model directory contains "
                "stock Whisper large-v3, not the Afrikaans fine-tuned merge. "
                "Stock Whisper will transcribe Afrikaans and produce fluent, "
                "plausible, subtly-wrong output — it is not safe for a legal or "
                "clinical transcript, so it will not be used."
            )
        else:
            headline = (
                "ACCURATE MODE REFUSED TO START: the model in this directory is "
                "not the approved Afrikaans merge, or is corrupt."
            )
        detail = "\n  - ".join(failures)
        raise AccurateModelGuardError(
            f"{headline}\n\nModel directory: {model_dir}\n"
            f"Failed checks:\n  - {detail}\n\n"
            f"Expected: {fingerprint.get('description', 'the approved merge')}\n"
            f"Fix: point STILLSCRIPT_ACCURATE_MODEL_DIR at the correct merged "
            f"model, or re-run verify_and_merge.py to rebuild it. If you "
            f"deliberately produced a new merge, re-pin it with "
            f"pin_accurate_fingerprint.py.",
            failures=failures, model_dir=model_dir,
        )

    if do_full:
        logger.info(
            "Accurate guard passed (FULL): %d probes + all %d tensors verified "
            "against the approved Afrikaans merge in %s",
            len(fingerprint["probes"]), tensors_hashed, model_dir)
    else:
        logger.info(
            "Accurate guard passed: %d probes verified against the approved "
            "Afrikaans merge in %s", len(fingerprint["probes"]), model_dir)
    return {
        "ok": True,
        "model_dir": model_dir,
        "probes_checked": len(fingerprint["probes"]),
        "tensor_count": len(key_to_file),
        "full": do_full,
        "tensors_hashed": tensors_hashed,
    }
