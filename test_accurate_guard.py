"""Tests for the Accurate-mode adapter guard (masterplan 2.2).

The guard's job is to refuse to run when the model on disk is not the approved
Afrikaans merge. So the important tests are the negative ones: it must actually
FAIL, with the right exception, on a stock/corrupt/wrong model. A guard that
only ever passes is indistinguishable from no guard at all.

Run with the spike venv (it has torch + safetensors):
    ~/whisper_afrikaans_spike/venv/bin/python3 test_accurate_guard.py

Most cases are synthetic and cost a few seconds. If the real base large-v3
checkpoint happens to be on this machine, one extra test runs the guard against
it directly — the strongest possible version of "would this catch a base model".
"""

import base64
import os
import shutil
import sys
import tempfile
import time
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


def expect_guard_error(label, model_dir, fingerprint, must_mention=None, full=None):
    """Assert the guard raises AccurateModelGuardError for this model dir."""
    try:
        accurate_guard.verify_merged_model(model_dir, fingerprint, full=full)
    except accurate_guard.AccurateModelGuardError as e:
        msg = str(e)
        if must_mention and must_mention.lower() not in msg.lower():
            return check(label, False,
                         f"raised, but message lacks {must_mention!r}: {msg[:200]}")
        first = (e.failures or ["<none>"])[0]
        return check(label, True, f"raised; first failure: {first[:90]}")
    except Exception as e:  # noqa: BLE001 - a wrong type here is itself a failure
        return check(label, False,
                     f"raised {type(e).__name__}, expected AccurateModelGuardError: {e}")
    return check(label, False, "guard PASSED a model it should have rejected")


print("\n=== 0. Import hygiene: the guard must not drag torch into Fast mode ===")
import accurate_guard  # noqa: E402
import accurate_engine  # noqa: E402
from accurate_fingerprint import ACCURATE_FINGERPRINT  # noqa: E402

check("importing accurate_guard does not import torch", "torch" not in sys.modules)
check("importing accurate_engine does not import torch", "torch" not in sys.modules)
check("importing accurate_engine does not import transformers",
      "transformers" not in sys.modules)

print("\n=== 1. Exception hierarchy ===")
check("AccurateModelGuardError subclasses AccurateEngineUnavailable",
      issubclass(accurate_guard.AccurateModelGuardError,
                 accurate_guard.AccurateEngineUnavailable))
check("AccurateModelGuardError is not a bare Exception subclass only",
      accurate_guard.AccurateModelGuardError is not Exception
      and issubclass(accurate_guard.AccurateModelGuardError, RuntimeError))
check("accurate_engine re-exports both error types",
      hasattr(accurate_engine, "AccurateModelGuardError")
      and hasattr(accurate_engine, "AccurateEngineUnavailable"))

print("\n=== 2. Fingerprint sanity ===")
probes = ACCURATE_FINGERPRINT["probes"]
finetuned = [p for p in probes if p["role"] == "finetuned"]
lineage = [p for p in probes if p["role"] == "lineage"]
check("fingerprint has finetuned probes", len(finetuned) >= 3, f"{len(finetuned)}")
check("fingerprint has lineage probes", len(lineage) >= 2, f"{len(lineage)}")
check("every finetuned probe is a DECODER tensor (encoder lora_B is all-zero)",
      all(".decoder.layers" in p["key"] for p in finetuned),
      ", ".join(p["key"].split("model.")[-1] for p in finetuned[:2]) + ", …")
check("no lineage probe is a q_proj/v_proj (a LoRA could move those)",
      all("q_proj" not in p["key"] and "v_proj" not in p["key"] for p in lineage))

print("\n=== 3. The real merged model must PASS ===")
real_dir = accurate_engine.resolve_model_dir()
print(f"  model dir: {real_dir}")
have_real = os.path.isfile(os.path.join(real_dir, "model.safetensors"))
if not have_real:
    check("real merged model present", False, real_dir)
    print("\nCannot continue without the merged model. Aborting.")
    sys.exit(1)

report = accurate_guard.verify_merged_model(real_dir)
check("guard passes against the real merged_afrikaans_fp32", report["ok"] is True)
check("guard checked every probe",
      report["probes_checked"] == len(probes), f"{report['probes_checked']}")

print("\n=== 4. The REAL stock large-v3 must FAIL (strongest negative test) ===")
base_dir = os.path.expanduser("~/whisper_afrikaans_spike/base_whisper_large_v3")
if os.path.isfile(os.path.join(base_dir, "model.safetensors")):
    expect_guard_error("stock base large-v3 is rejected", base_dir,
                       ACCURATE_FINGERPRINT, must_mention="stock Whisper")
else:
    print("  [SKIP] base large-v3 not on this machine; synthetic stand-in below")

# ── Synthetic models ────────────────────────────────────────────────────────
# Built from the pinned base samples, so they work on any machine without a
# 6 GB download. Each holds only the 9 probe tensors at full shape; the test
# fingerprint's tensor_count is adjusted to match so that the structural check
# passes and the interesting check is the one under test.
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

TMP = tempfile.mkdtemp(prefix="guard_test_", dir=os.environ.get(
    "TMPDIR", tempfile.gettempdir()))


def base_sample_of(probe):
    return torch.frombuffer(
        bytearray(base64.b64decode(probe["base_sample_b64"])), dtype=torch.float32)


def synth_fingerprint(n_tensors=None):
    fp = copy.deepcopy(ACCURATE_FINGERPRINT)
    fp["tensor_count"] = n_tensors if n_tensors is not None else len(fp["probes"])
    return fp


def make_model(name, values, skip=(), dtype=torch.float32):
    """Write a synthetic model dir. `values(probe, base_sample)` returns the
    sample values to plant at the probe's sampled indices (rest stays zero)."""
    d = os.path.join(TMP, name)
    os.makedirs(d, exist_ok=True)
    tensors = {}
    for p in ACCURATE_FINGERPRINT["probes"]:
        if p["key"] in skip:
            continue
        bs = base_sample_of(p)
        flat = torch.zeros(int(torch.tensor(p["shape"]).prod()), dtype=torch.float32)
        idx = accurate_guard._sample_indices(flat.numel(), bs.numel())
        flat[idx] = values(p, bs)
        tensors[p["key"]] = flat.reshape(p["shape"]).to(dtype).contiguous()
    save_file(tensors, os.path.join(d, "model.safetensors"))
    return d


print("\n=== 5. Synthetic stand-in for a stock/base model must FAIL ===")
# Values at the probed positions are exactly base large-v3's values => the
# guard should see zero delta on the finetuned probes and name it as stock.
d = make_model("as_base", lambda p, bs: bs)
expect_guard_error("synthetic base stand-in is rejected", d, synth_fingerprint(),
                   must_mention="stock Whisper")

print("\n=== 6. A barely-changed model must FAIL (below the delta threshold) ===")
# Simulates an adapter that trained to almost nothing, or a merge that applied
# a near-zero LoRA: not bitwise-base, but not meaningfully fine-tuned either.
d = make_model("barely", lambda p, bs: bs * (1 + 1e-6) if p["role"] == "finetuned" else bs)
expect_guard_error("near-zero fine-tuning is rejected", d, synth_fingerprint(),
                   must_mention="un-fine-tuned")

print("\n=== 7. A model with the wrong lineage must FAIL ===")
# Finetuned probes look adapted, but a tensor no q_proj/v_proj LoRA could ever
# touch has moved => this is not large-v3 + Afrikaans adapter.
d = make_model("wrong_lineage",
               lambda p, bs: bs * 1.5 if p["role"] == "lineage" else bs * 1.2)
expect_guard_error("altered lineage tensor is rejected", d, synth_fingerprint(),
                   must_mention="cannot touch this tensor")

print("\n=== 8. Corruption must FAIL ===")
d = make_model("nan", lambda p, bs: bs * float("nan") if p["role"] == "finetuned" else bs)
expect_guard_error("NaN weights are rejected", d, synth_fingerprint(),
                   must_mention="NaN")

d = make_model("truncated", lambda p, bs: bs * 1.2,
               skip=(finetuned[0]["key"], finetuned[1]["key"]))
expect_guard_error("missing tensors (partial download) are rejected", d,
                   synth_fingerprint(len(probes) - 2), must_mention="missing")

print("\n=== 9. A re-quantised model must FAIL ===")
d = make_model("fp16", lambda p, bs: bs * 1.2, dtype=torch.float16)
expect_guard_error("fp16 re-save is rejected", d, synth_fingerprint(),
                   must_mention="dtype")

print("\n=== 10. Structural check: wrong tensor count must FAIL ===")
expect_guard_error("unexpected tensor count is rejected", real_dir,
                   synth_fingerprint(9999), must_mention="tensor count")

print("\n=== 11. Empty / missing weights must FAIL ===")
empty = os.path.join(TMP, "empty")
os.makedirs(empty, exist_ok=True)
expect_guard_error("directory with no weights is rejected", empty,
                   synth_fingerprint(), must_mention="no safetensors weights")

print("\n=== 12. Identity check: a different-but-plausible merge must FAIL ===")
# The real model, but the pinned sha256 says something else — i.e. someone
# swapped in a different fine-tune that is still genuinely adapted large-v3.
# Evidence checks all pass; only the identity check can catch this.
tampered = copy.deepcopy(ACCURATE_FINGERPRINT)
tampered["probes"][0]["sha256"] = "0" * 64
expect_guard_error("unrecognised merge is rejected", real_dir,
                   tampered, must_mention="does not match the approved merge")

print("\n=== 13. There is NO runtime bypass ===")
# The guard must not be weakenable from the environment. The supported path for
# a deliberate re-merge is to re-pin with pin_accurate_fingerprint.py, which is
# a reviewable change to a committed file — not a runtime toggle.
for gone in ("guard_mode", "GUARD_MODE_ENV_VAR", "EVIDENCE_ONLY", "STRICT"):
    check(f"accurate_guard no longer exposes {gone}",
          not hasattr(accurate_guard, gone))
check("the guard report no longer carries a mode/tolerated field",
      "mode" not in report and "tolerated" not in report, str(sorted(report)))

# Setting the retired variable (or anything like it) must change nothing.
bypass_attempts = {
    "STILLSCRIPT_ACCURATE_GUARD": "evidence",
    "STILLSCRIPT_ACCURATE_GUARD_MODE": "off",
    "STILLSCRIPT_SKIP_GUARD": "1",
}
os.environ.update(bypass_attempts)
try:
    expect_guard_error("sha mismatch is still fatal with bypass vars set",
                       real_dir, tampered,
                       must_mention="does not match the approved merge")
    d = make_model("as_base2", lambda p, bs: bs)
    expect_guard_error("stock base model is still rejected with bypass vars set",
                       d, synth_fingerprint(), must_mention="stock Whisper")
    check("the approved model still passes with those vars set",
          accurate_guard.verify_merged_model(real_dir)["ok"] is True)
finally:
    for k in bypass_attempts:
        os.environ.pop(k, None)

print("\n=== 14. The guard runs once, at load time, BEFORE the 6 GB load ===")
# Stub out from_pretrained so this costs no time, and record call order.
from transformers import WhisperProcessor, WhisperForConditionalGeneration  # noqa: E402

order = []
real_verify = accurate_engine.verify_merged_model
real_proc_fp = WhisperProcessor.from_pretrained
real_model_fp = WhisperForConditionalGeneration.from_pretrained


def spy_verify(model_dir, *a, **k):
    order.append("guard")
    return real_verify(model_dir, *a, **k)


accurate_engine.verify_merged_model = spy_verify
WhisperProcessor.from_pretrained = classmethod(
    lambda cls, *a, **k: order.append("load_processor") or object())
WhisperForConditionalGeneration.from_pretrained = classmethod(
    lambda cls, *a, **k: order.append("load_model") or type(
        "M", (), {"eval": lambda self: None})())
try:
    accurate_engine._engine_cache.pop(real_dir, None)
    accurate_engine.load_engine(real_dir)
    accurate_engine.load_engine(real_dir)   # second call must hit the cache
    accurate_engine.load_engine(real_dir)   # and so must the third
finally:
    accurate_engine.verify_merged_model = real_verify
    WhisperProcessor.from_pretrained = real_proc_fp
    WhisperForConditionalGeneration.from_pretrained = real_model_fp
    accurate_engine._engine_cache.pop(real_dir, None)

check("guard ran exactly once across three load_engine() calls",
      order.count("guard") == 1, f"order={order}")
check("guard ran BEFORE the model was loaded",
      order and order[0] == "guard", f"order={order}")
check("model was loaded exactly once", order.count("load_model") == 1)

print("\n=== 15. A guard failure stops load_engine (no fallback) ===")
accurate_engine._engine_cache.pop(real_dir, None)


def failing_verify(model_dir, *a, **k):
    raise accurate_guard.AccurateModelGuardError("simulated wrong model", ["boom"], model_dir)


accurate_engine.verify_merged_model = failing_verify
loaded_anyway = False
try:
    accurate_engine.load_engine(real_dir)
    loaded_anyway = True
except accurate_guard.AccurateModelGuardError:
    pass
except accurate_engine.AccurateEngineUnavailable:
    pass  # also acceptable: same hierarchy
finally:
    accurate_engine.verify_merged_model = real_verify
    accurate_engine._engine_cache.pop(real_dir, None)

check("load_engine propagates the guard error instead of loading anyway",
      not loaded_anyway)
check("nothing was cached after a guard failure",
      real_dir not in accurate_engine._engine_cache)

print("\n=== 16. Opt-in full verification ===")
check("full mode is OFF by default", report.get("full") is False, str(report.get("full")))
check("default mode hashes no whole-file manifest",
      report.get("tensors_hashed") == 0)
check("a full_manifest is pinned",
      len(ACCURATE_FINGERPRINT.get("full_manifest", {})) == report["tensor_count"],
      f"{len(ACCURATE_FINGERPRINT.get('full_manifest', {}))} digests")

t0 = time.time()
full_report = accurate_guard.verify_merged_model(real_dir, full=True)
full_secs = time.time() - t0
check("(a) full mode passes on the real approved merge", full_report["ok"] is True)
check("full mode hashed every tensor",
      full_report["tensors_hashed"] == full_report["tensor_count"],
      f"{full_report['tensors_hashed']} tensors in {full_secs:.2f}s")
check("full mode still ran the probe checks too",
      full_report["probes_checked"] == len(probes))

# (c) opt-in: resolution order must match resolve_model_dir() — arg, then env.
check("full_verify_requested() defaults to False", not accurate_guard.full_verify_requested())
os.environ[accurate_guard.FULL_VERIFY_ENV_VAR] = "1"
try:
    check("env var turns full mode on", accurate_guard.full_verify_requested())
    check("explicit full=False beats the env var",
          not accurate_guard.full_verify_requested(full=False))
    env_report = accurate_guard.verify_merged_model(real_dir)
    check("env var reaches verify_merged_model", env_report["full"] is True)
finally:
    del os.environ[accurate_guard.FULL_VERIFY_ENV_VAR]
os.environ[accurate_guard.FULL_VERIFY_ENV_VAR] = "no"
try:
    check("a non-truthy env value leaves full mode off",
          not accurate_guard.full_verify_requested())
finally:
    del os.environ[accurate_guard.FULL_VERIFY_ENV_VAR]
check("explicit full=True beats an unset env var",
      accurate_guard.full_verify_requested(full=True))

# (c) default behaviour and timing must be untouched by all of the above.
t0 = time.time()
again = accurate_guard.verify_merged_model(real_dir)
default_secs = time.time() - t0
check("default report is unchanged after full-mode work",
      again["ok"] is True and again["full"] is False and again["tensors_hashed"] == 0)
check("default stays far cheaper than full",
      default_secs < full_secs / 2,
      f"default {default_secs:.2f}s vs full {full_secs:.2f}s")

print("\n=== 17. (b) full mode still rejects the bad models ===")
# Synthetic dirs hold only the 9 probe tensors, so their manifests are the 9
# corresponding pinned digests — full mode must reject them on content too.
synth_full = copy.deepcopy(synth_fingerprint())
synth_full["full_manifest"] = {
    p["key"]: ACCURATE_FINGERPRINT["full_manifest"][p["key"]] for p in probes}

for name, values, mention in (
    ("full_as_base", lambda p, bs: bs, "stock Whisper"),
    ("full_lineage", lambda p, bs: bs * 1.5 if p["role"] == "lineage" else bs * 1.2,
     "cannot touch this tensor"),
    ("full_nan", lambda p, bs: bs * float("nan") if p["role"] == "finetuned" else bs,
     "NaN"),
):
    d = make_model(name, values)
    expect_guard_error(f"full mode rejects {name}", d, synth_full,
                       must_mention=mention, full=True)

# A tensor the probes do NOT cover is corrupt: default passes, full catches it.
# This is the whole reason full mode exists, so it is the important case.
unprobed_key = sorted(
    k for k in ACCURATE_FINGERPRINT["full_manifest"]
    if k not in {p["key"] for p in probes} and "layers.7." in k)[0]
print(f"  corrupting an unprobed tensor: {unprobed_key}")
corrupt_dir = os.path.join(TMP, "unprobed_corrupt")
os.makedirs(corrupt_dir, exist_ok=True)
tensors = {}
for p in ACCURATE_FINGERPRINT["probes"]:
    src = accurate_guard._weight_files(real_dir)[0]
    from safetensors import safe_open as _so
    tensors[p["key"]] = _so(src, framework="pt").get_tensor(p["key"])
tensors[unprobed_key] = torch.zeros(8, dtype=torch.float32)  # wrong content+shape
save_file(tensors, os.path.join(corrupt_dir, "model.safetensors"))

partial_fp = copy.deepcopy(ACCURATE_FINGERPRINT)
partial_fp["tensor_count"] = len(tensors)
partial_fp["full_manifest"] = {k: ACCURATE_FINGERPRINT["full_manifest"][k] for k in tensors}

ok_default = accurate_guard.verify_merged_model(corrupt_dir, partial_fp)
check("default mode PASSES a file whose corruption is outside the probes",
      ok_default["ok"] is True, "this is precisely the gap full mode closes")
expect_guard_error("full mode CATCHES corruption outside the probes",
                   corrupt_dir, partial_fp, must_mention="full check", full=True)

print("\n=== 18. Full mode needs a pinned manifest ===")
no_manifest = copy.deepcopy(ACCURATE_FINGERPRINT)
no_manifest.pop("full_manifest")
try:
    accurate_guard.verify_merged_model(real_dir, no_manifest, full=True)
    check("missing full_manifest is a loud failure", False, "guard passed")
except accurate_guard.AccurateModelGuardError as e:
    check("missing full_manifest is a loud failure", "full_manifest" in str(e))
check("...but the same fingerprint still works in default mode",
      accurate_guard.verify_merged_model(real_dir, no_manifest)["ok"] is True)

shutil.rmtree(TMP, ignore_errors=True)

print("\n" + "=" * 68)
if failures:
    print(f"RESULT: FAIL — {len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("RESULT: PASS — guard accepts the approved merge and rejects stock,\n"
      "        barely-tuned, wrong-lineage, corrupt, truncated, re-quantised\n"
      "        and unrecognised models; runs once, before the model load.")
