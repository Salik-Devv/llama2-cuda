#!/usr/bin/env python3
"""
analyze_profile.py  –  Parse llama_profile.csv and produce bottleneck charts.

Usage:
    python3 src/analyze_profile.py results/llama_profile.csv
"""

import sys
import csv
import os
import statistics

def load_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if k != 'pos' else int(v) for k, v in row.items()})
    return rows

def print_banner(title):
    w = 72
    print("=" * w)
    print(f"  {title}")
    print("=" * w)

def bar(label, value, total, width=40):
    frac = value / total if total > 0 else 0
    filled = int(frac * width)
    bar_str = "█" * filled + "░" * (width - filled)
    return f"  {label:<22} {bar_str}  {frac*100:5.1f}%  ({value:.4f} ms)"

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <profile.csv>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}"); sys.exit(1)

    rows = load_csv(path)
    if not rows:
        print("Empty profile CSV."); sys.exit(1)

    n = len(rows)

    # Per-token averages
    keys = ['kernel_ms', 'bw_GBs',
            'matmul_ms_per_layer', 'rmsnorm_ms_per_layer',
            'rope_ms_per_layer', 'attn_ms_per_layer', 'swiglu_ms_per_layer']

    avg = {k: statistics.mean(r[k] for r in rows) for k in keys}
    med = {k: statistics.median(r[k] for r in rows) for k in keys}
    mn  = {k: min(r[k] for r in rows) for k in keys}
    mx  = {k: max(r[k] for r in rows) for k in keys}

    print_banner("LLaMA-2 CUDA Bottleneck Analysis")
    print(f"\n  Tokens profiled : {n}")
    print(f"  Avg kernel time : {avg['kernel_ms']:.4f} ms / token")
    print(f"  Median          : {med['kernel_ms']:.4f} ms")
    print(f"  Min / Max       : {mn['kernel_ms']:.4f} / {mx['kernel_ms']:.4f} ms")
    print(f"  Effective BW    : {avg['bw_GBs']:.2f} GB/s  (avg)")

    print(f"\n{'─'*72}")
    print("  Per-Layer Operation Breakdown (% of total kernel time)")
    print(f"{'─'*72}")

    ops = [
        ('matmul_ms_per_layer',   'MatMul (cuBLAS)'),
        ('attn_ms_per_layer',     'Multi-Head Attn'),
        ('rmsnorm_ms_per_layer',  'RMSNorm'),
        ('swiglu_ms_per_layer',   'SwiGLU'),
        ('rope_ms_per_layer',     'RoPE'),
    ]

    total_accounted = sum(avg[k] for k, _ in ops)
    for k, label in sorted(ops, key=lambda x: -avg[x[0]]):
        print(bar(label, avg[k], total_accounted))

    print(f"\n  (Remaining: other ops incl. residual adds, KV cache writes)")

    print(f"\n{'─'*72}")
    print("  Token-level Latency Statistics")
    print(f"{'─'*72}")
    print(f"  {'Metric':<28} {'Mean':>9}  {'Median':>9}  {'Min':>9}  {'Max':>9}")
    print(f"  {'──────':<28} {'────':>9}  {'──────':>9}  {'───':>9}  {'───':>9}")
    for k in keys:
        unit = "GB/s" if k == 'bw_GBs' else "ms"
        label = k.replace('_ms_per_layer','').replace('_ms','').replace('_',' ').title()
        print(f"  {label:<28} {avg[k]:>8.4f}  {med[k]:>8.4f}  {mn[k]:>8.4f}  {mx[k]:>8.4f}  {unit}")

    # Bottleneck identification
    top_op = max(ops, key=lambda x: avg[x[0]])
    print(f"\n{'─'*72}")
    print(f"  ⚑  Primary bottleneck: {top_op[1]}  ({100*avg[top_op[0]]/total_accounted:.1f}% of compute)")
    if avg['bw_GBs'] < 200:
        print(f"  ⚑  Memory bandwidth appears sub-optimal ({avg['bw_GBs']:.1f} GB/s)")
        print(f"     → Consider FP16 (half2) matmuls or weight quantization (INT8)")
    else:
        print(f"  ✓  Memory bandwidth looks healthy ({avg['bw_GBs']:.1f} GB/s)")

    if avg['attn_ms_per_layer'] / total_accounted > 0.25:
        print(f"  ⚑  Attention cost is high — consider FlashAttention or grouped-query attn")

    print("=" * 72)

    # Optional: write summary CSV
    out = os.path.join(os.path.dirname(path), "bottleneck_summary.csv")
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['op', 'avg_ms', 'median_ms', 'min_ms', 'max_ms', 'pct'])
        for k, label in ops:
            w.writerow([label,
                        f"{avg[k]:.4f}", f"{med[k]:.4f}",
                        f"{mn[k]:.4f}",  f"{mx[k]:.4f}",
                        f"{100*avg[k]/total_accounted:.2f}"])
    print(f"\n  Summary CSV: {out}")

if __name__ == '__main__':
    main()