"""Regenerate the pinned fingerprint used by accurate_engine's adapter guard.

Run this ONLY when you have deliberately produced a new merged model (e.g. a
re-merge, or a newer Afrikaans adapter) and have satisfied yourself that the
new model is correct. It rewrites the ACCURATE_FINGERPRINT block in
accurate_guard.py, which is the thing that decides whether Accurate mode is
allowed to run at all.

It needs BOTH the merged model and the base large-v3 checkpoint, because the
fingerprint records, for each probe tensor, what the base model's values are —
that is what lets the guard say "you have loaded stock Whisper" rather than
just "hash mismatch".

    ~/whisper_afrikaans_spike/venv/bin/python3 pin_accurate_fingerprint.py \
        --merged ~/whisper_afrikaans_spike/merged_afrikaans_fp32 \
        --base   ~/whisper_afrikaans_spike/base_whisper_large_v3

Print-only (show what would change, write nothing):

    ... pin_accurate_fingerprint.py --check
"""

import argparse
import base64
import hashlib
import json
import os
import sys
import time

# The probe tensors. Chosen from the ground truth established against André
# Oosthuizen's adapter (LoRA r=32, alpha=64, target_modules=[q_proj, v_proj]):
#
#   * The adapter carries LoRA tensors for the encoder too, but every encoder
#     lora_B is still exactly zero — i.e. untrained. Merging therefore leaves
#     all 64 encoder q/v_proj weights BIT-IDENTICAL to base large-v3.
#     Only the 128 decoder q/v_proj tensors actually move.
#     => "finetuned" probes must all be decoder tensors. Sampling an encoder
#        q_proj here would look exactly like a stock model and permanently
#        false-alarm.
#
#   * "lineage" probes are tensors that a q_proj/v_proj LoRA can never touch,
#     so they must still match base large-v3 exactly. They prove the file is
#     still genuinely large-v3 and not some other model that happens to differ
#     from base in the decoder.
FINETUNED_PROBES = [
    "model.decoder.layers.0.self_attn.q_proj.weight",
    "model.decoder.layers.0.encoder_attn.v_proj.weight",
    "model.decoder.layers.15.self_attn.v_proj.weight",
    "model.decoder.layers.15.encoder_attn.q_proj.weight",
    "model.decoder.layers.31.self_attn.q_proj.weight",
    "model.decoder.layers.31.encoder_attn.v_proj.weight",
]
LINEAGE_PROBES = [
    "model.encoder.conv1.weight",
    "model.encoder.layers.0.self_attn.k_proj.weight",
    "model.decoder.layers.0.final_layer_norm.weight",
]

# How many values to record per probe. These are compared elementwise against
# the merged model at guard time, so they need only be enough to make an
# accidental match impossible — 64 float32 values is already astronomically
# more than enough, and keeps the pinned constant ~3 KB.
SAMPLE_N = 64


def sample_indices(numel, n=SAMPLE_N):
    """Deterministic evenly-spread indices into a flattened tensor."""
    n = min(n, numel)
    step = numel / n
    return [min(numel - 1, int(i * step)) for i in range(n)]


def probe_record(merged_f, base_f, key, role):
    import torch

    tm = merged_f.get_tensor(key)
    tb = base_f.get_tensor(key)
    if tuple(tm.shape) != tuple(tb.shape):
        raise SystemExit(f"shape mismatch between merged and base for {key}")

    flat_m = tm.reshape(-1).to(dtype=torch.float32)
    flat_b = tb.reshape(-1).to(dtype=torch.float32)
    idx = sample_indices(flat_m.numel())

    base_sample = flat_b[idx].contiguous()
    merged_sample = flat_m[idx].contiguous()

    # sha256 over the merged tensor's raw bytes: exact identity of the file we
    # validated. Computed on the float32 view so it is independent of how
    # safetensors happened to lay the tensor out.
    digest = hashlib.sha256(flat_m.numpy().tobytes()).hexdigest()

    rel = float((merged_sample - base_sample).norm() / (base_sample.norm() + 1e-12))
    return {
        "key": key,
        "role": role,
        "shape": list(tm.shape),
        "dtype": str(tm.dtype).replace("torch.", ""),
        "sha256": digest,
        "base_sample_b64": base64.b64encode(base_sample.numpy().tobytes()).decode(),
    }, rel


def build(merged_dir, base_dir):
    from safetensors import safe_open

    mpath = os.path.join(merged_dir, "model.safetensors")
    bpath = os.path.join(base_dir, "model.safetensors")
    for p in (mpath, bpath):
        if not os.path.isfile(p):
            raise SystemExit(f"not found: {p}")

    merged_f = safe_open(mpath, framework="pt")
    base_f = safe_open(bpath, framework="pt")

    all_keys = list(merged_f.keys())
    probes = []
    # Imported here, not at module scope, so this script can still run when
    # accurate_fingerprint.py is absent (bootstrap / regenerate-from-scratch).
    from accurate_guard import full_tensor_digest
    print(f"{'probe':58s} {'role':10s} {'rel delta vs base':>18s}")
    for key, role in (
        [(k, "finetuned") for k in FINETUNED_PROBES]
        + [(k, "lineage") for k in LINEAGE_PROBES]
    ):
        if key not in all_keys:
            raise SystemExit(f"probe key absent from merged model: {key}")
        rec, rel = probe_record(merged_f, base_f, key, role)
        probes.append(rec)
        print(f"{key:58s} {role:10s} {rel:18.6e}")
        if role == "finetuned" and rel == 0.0:
            raise SystemExit(
                f"\nREFUSING TO PIN: probe {key} is identical to base large-v3.\n"
                "The merged model you pointed at is not fine-tuned there."
            )
        if role == "lineage" and rel != 0.0:
            raise SystemExit(
                f"\nREFUSING TO PIN: lineage probe {key} differs from base.\n"
                "This model is not a plain q_proj/v_proj LoRA merge of large-v3."
            )

    # Full manifest: one digest per tensor, for opt-in full verification.
    # Generated unconditionally and in the same pass as the probes so the two
    # can never describe different models.
    print(f"\nhashing all {len(all_keys)} tensors for the full manifest "
          f"(reads the whole file, takes a few seconds) ...")
    t0 = time.time()
    manifest = {k: full_tensor_digest(k, merged_f.get_tensor(k))
                for k in sorted(all_keys)}
    print(f"full manifest: {len(manifest)} digests in {time.time() - t0:.1f}s")

    return {
        "schema": 2,
        "description": (
            "Whisper large-v3 + André Oosthuizen Afrikaans LoRA (r=32, alpha=64, "
            "target_modules=[q_proj, v_proj]) merged to fp32 via merge_and_unload."
        ),
        "tensor_count": len(all_keys),
        "probes": probes,
        "full_manifest": manifest,
    }


def render(fp):
    body = json.dumps(fp, indent=4, sort_keys=False)
    body = "\n".join(("    " + ln) if ln.strip() else ln for ln in body.splitlines())
    return "ACCURATE_FINGERPRINT = " + body.lstrip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=os.path.expanduser(
        "~/whisper_afrikaans_spike/merged_afrikaans_fp32"))
    ap.add_argument("--base", default=os.path.expanduser(
        "~/whisper_afrikaans_spike/base_whisper_large_v3"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "accurate_fingerprint.py"))
    ap.add_argument("--check", action="store_true",
                    help="print the fingerprint, write nothing")
    args = ap.parse_args()

    fp = build(args.merged, args.base)
    text = (
        '"""Pinned fingerprint of the approved merged Afrikaans model.\n\n'
        "GENERATED FILE — do not hand-edit. Regenerate with:\n"
        "    python3 pin_accurate_fingerprint.py\n"
        "and only after you have deliberately produced and checked a new merge.\n"
        "Changing this file changes which model Accurate mode will accept.\n"
        '"""\n\n' + render(fp)
    )
    if args.check:
        sys.stdout.write("\n" + text)
        return
    with open(args.out, "w") as fh:
        fh.write(text)
    print(f"\nwrote {args.out}  ({len(fp['probes'])} probes, "
          f"tensor_count={fp['tensor_count']})")


if __name__ == "__main__":
    main()
