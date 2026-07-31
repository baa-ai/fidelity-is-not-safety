#!/usr/bin/env python
"""coherence probe — the data-free detector's two axes, measured on any model.

From "Fidelity Is Not Safety" (arXiv:2607.28196). The agentic failure is governed by the
COHERENCE of the compression error times its RATE, not by the amount of damage. This probe
applies each compression operator (low-rank SVD truncation, magnitude pruning, quantization)
to a model at a grid of doses and measures two data-free statistics of the error
dW = W_base - W_compressed, aggregated (energy-weighted) over the linear tensors:

  coherent_fraction = sum_t topk_energy(dW_t) / sum_t ||dW_t||_F^2   (k=8; is the error low-rank?)
  error_rate        = sum_t ||dW_t||_F^2      / sum_t ||W_t||_F^2     (how large is the dose?)

The screen FLAGs a build iff coherent_fraction > 0.007 AND error_rate > 0.01 (paper thresholds,
calibrated on 7-8B dense models — advisory, recalibrate per family). You will see low-rank SVD
sit ABOVE the coherence gate at every dose while pruning stays BELOW it at every density, which
is why coherent SVD invents procedure steps and damage-matched pruning does not.

Usage:
  python coherence_probe.py --model /path/to/bf16-model
  python coherence_probe.py --model ... --ops svd prune quant
  python coherence_probe.py --model ... --max-tensors 60      # faster (sample tensors)

Requires: mlx, mlx-lm, numpy (Apple Silicon). No data, no generation — pure linear algebra.
"""
import argparse, json, sys
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load

COH_THRESH, RATE_THRESH, K = 0.007, 0.01, 8


def is_linear(name, shape):
    low = name.lower()
    return (name.endswith(".weight") and len(shape) == 2 and min(shape) >= 512
            and ".layers." in low and "embed" not in low and "norm" not in low
            and "lm_head" not in low)


def topk_energy(dW, k=K, iters=3):
    """squared top-k singular values via randomized subspace iteration (float32)."""
    m, n = dW.shape
    kk = max(1, min(k, min(m, n) - 1))
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((n, kk)).astype(np.float32)
    for _ in range(iters):
        Q, _ = np.linalg.qr(dW @ Q)
        Q, _ = np.linalg.qr(dW.T @ Q)
    s = np.linalg.svd(dW @ Q, compute_uv=False).astype(np.float64)
    return float((s ** 2).sum())


def rank_at(frac, m, n):
    return max(1, min(min(m, n), int(frac * m * n / (m + n))))


def svd_arm(S, WE, budget, m, n):
    """SVD-truncation error stats from the singular spectrum S (no reconstruction needed):
    the discarded tail's singular values ARE S[r:], so top-k energy = S[r:r+K]^2."""
    r = rank_at(budget, m, n)
    tail = S[r:]
    if tail.size == 0:
        return 0.0, 0.0
    dE = float((tail.astype(np.float64) ** 2).sum())
    topk = float((tail[:K].astype(np.float64) ** 2).sum())
    return topk, dE


def prune_arm(W, density):
    n = W.size
    kth = max(0, min(n - 1, n - int(density * n)))
    thr = np.partition(np.abs(W).reshape(-1), kth)[kth]
    dW = np.where(np.abs(W) < thr, W, 0.0).astype(np.float32)
    return dW


def quant_arm(Wm, bits, gs=64):
    wq, sc, bi = mx.quantize(Wm, group_size=gs, bits=bits)
    deq = mx.dequantize(wq, sc, bi, group_size=gs, bits=bits).astype(mx.float32)
    return np.array(deq - Wm.astype(mx.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="bf16 model (HF id or local path)")
    ap.add_argument("--ops", nargs="+", default=["svd", "prune", "quant"],
                    choices=["svd", "prune", "quant"])
    ap.add_argument("--svd-budgets", type=float, nargs="+", default=[0.99, 0.98, 0.97, 0.95])
    ap.add_argument("--prune-densities", type=float, nargs="+", default=[0.85, 0.75, 0.65, 0.55])
    ap.add_argument("--quant-bits", type=int, nargs="+", default=[4, 3, 2])
    ap.add_argument("--max-tensors", type=int, default=None, help="sample N tensors (faster)")
    ap.add_argument("--output", default=None)
    a = ap.parse_args()

    print(f"loading {a.model} ...", flush=True)
    model, _ = load(a.model)
    params = dict(tree_flatten(model.parameters()))
    names = [n for n in params if is_linear(n, tuple(params[n].shape))]
    if a.max_tensors and a.max_tensors < len(names):
        step = len(names) / a.max_tensors
        names = [names[int(i * step)] for i in range(a.max_tensors)]
    print(f"probing {len(names)} linear tensors with ops={a.ops} ... (pure linear algebra, no data)",
          flush=True)

    # accumulators: {(op, dose): [topk, dE, WE]}
    acc = {}
    for idx, nm in enumerate(names):
        Wm = params[nm].astype(mx.float32)
        W = np.array(Wm)
        m, n = W.shape
        WE = float((W.astype(np.float64) ** 2).sum())
        if "svd" in a.ops:
            S = np.linalg.svd(W, compute_uv=False).astype(np.float32)
            for b in a.svd_budgets:
                topk, dE = svd_arm(S, WE, b, m, n)
                k = ("svd", b); acc.setdefault(k, [0.0, 0.0, 0.0])
                acc[k][0] += topk; acc[k][1] += dE; acc[k][2] += WE
        if "prune" in a.ops:
            for d in a.prune_densities:
                dW = prune_arm(W, d); dE = float((dW.astype(np.float64) ** 2).sum())
                topk = topk_energy(dW) if dE > 0 else 0.0
                k = ("prune", d); acc.setdefault(k, [0.0, 0.0, 0.0])
                acc[k][0] += topk; acc[k][1] += dE; acc[k][2] += WE
        if "quant" in a.ops:
            for bits in a.quant_bits:
                dW = quant_arm(Wm, bits); dE = float((dW.astype(np.float64) ** 2).sum())
                topk = topk_energy(dW) if dE > 0 else 0.0
                k = ("quant", bits); acc.setdefault(k, [0.0, 0.0, 0.0])
                acc[k][0] += topk; acc[k][1] += dE; acc[k][2] += WE
        if (idx + 1) % 40 == 0:
            print(f"  {idx+1}/{len(names)} tensors", flush=True)

    rows = []
    for (op, dose), (topk, dE, WE) in acc.items():
        cf = topk / dE if dE else 0.0
        er = dE / WE if WE else 0.0
        flag = "FLAG" if (cf > COH_THRESH and er > RATE_THRESH) else "pass"
        rows.append({"op": op, "dose": dose, "coherent_fraction": cf, "error_rate": er, "gate": flag})
    rows.sort(key=lambda r: (r["op"], -r["dose"] if r["op"] != "quant" else r["dose"]))

    print(f"\n=== coherence probe: {a.model} ===")
    print(f"  gate: FLAG iff coherent_fraction > {COH_THRESH} AND error_rate > {RATE_THRESH}\n")
    print(f"  {'operator':10} {'dose':>7}  {'coherent_fraction':>17}  {'error_rate':>11}   gate")
    for r in rows:
        dose = f"b={r['dose']}" if r["op"] == "svd" else (f"d={r['dose']}" if r["op"] == "prune" else f"{r['dose']}bit")
        print(f"  {r['op']:10} {dose:>7}  {r['coherent_fraction']:>17.4f}  {r['error_rate']:>11.4f}   {r['gate']}")
    print("\n  Read it: low-rank SVD sits above the coherence gate (0.007) at every dose;")
    print("  pruning stays below it at every density. Coherence x rate, not damage.")
    if a.output:
        json.dump({"model": a.model, "rows": rows}, open(a.output, "w"), indent=2)
        print(f"  wrote {a.output}")


if __name__ == "__main__":
    main()
