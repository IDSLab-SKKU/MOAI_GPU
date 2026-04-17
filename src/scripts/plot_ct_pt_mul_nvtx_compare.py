#!/usr/bin/env python3
"""
Compare CT×PT micro-benchmark kernel summaries (mul_nvtx clean TSV) across runs
and emit:
  - total-time bar chart
  - stacked breakdown chart (top kernels)
  - JSON summary (totals + speedups)

Input TSV format: produced by export_ct_pt_kern_sum.py (tab-separated with at least
Name/ShortName/Total Time (ns)).
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Run:
    label: str
    path: Path
    totals_ns_by_kernel: Dict[str, float]

    @property
    def total_ns(self) -> float:
        return float(sum(self.totals_ns_by_kernel.values()))


def _read_clean_tsv(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        out: Dict[str, float] = {}
        for row in r:
            k = (row.get("ShortName") or row.get("Name") or "").strip()
            v = (row.get("Total Time (ns)") or "").replace(",", "").strip()
            if not k or not v:
                continue
            try:
                out[k] = out.get(k, 0.0) + float(v)
            except ValueError:
                continue
    return out


def _auto_label(path: Path) -> str:
    name = path.name
    # ct_pt_micro_v2_scalar_only_run_run1.kern_sum.mul_nvtx_clean.tsv -> v2_scalar_only
    if name.startswith("ct_pt_micro_") and ".kern_sum" in name:
        core = name[len("ct_pt_micro_") : name.index(".kern_sum")]
        core = core.replace("_run_run", "_run").replace("_run1", "").replace("_run2", "")
        return core
    return path.stem


def _pick_top_kernels(runs: List[Run], top_n: int) -> List[str]:
    agg: Dict[str, float] = {}
    for run in runs:
        for k, v in run.totals_ns_by_kernel.items():
            agg[k] = agg.get(k, 0.0) + float(v)
    return [k for k, _ in sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]


def _ms(ns: float) -> float:
    return float(ns) / 1e6


def _plot_total_bar(out_png: Path, runs: List[Run]) -> None:
    import matplotlib.pyplot as plt

    labels = [r.label for r in runs]
    vals_ms = [_ms(r.total_ns) for r in runs]

    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(runs) + 2.0), 4.2))
    ax.bar(labels, vals_ms, color="#4472c4")
    ax.set_ylabel("Total GPU time in mul_nvtx (ms)")
    ax.set_title("CT×PT mul_nvtx total time (kernel sum)")
    for i, v in enumerate(vals_ms):
        ax.text(i, v, f"{v:.3f} ms", ha="center", va="bottom", fontsize=9, rotation=0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _plot_breakdown_stacked(out_png: Path, runs: List[Run], top_kernels: List[str]) -> None:
    import matplotlib.pyplot as plt

    labels = [r.label for r in runs]
    other_ms: List[float] = []
    series: List[Tuple[str, List[float]]] = []
    for k in top_kernels:
        series.append((k, [_ms(r.totals_ns_by_kernel.get(k, 0.0)) for r in runs]))

    for r in runs:
        top_sum = sum(r.totals_ns_by_kernel.get(k, 0.0) for k in top_kernels)
        other_ms.append(_ms(max(0.0, r.total_ns - top_sum)))

    bottoms = [0.0] * len(runs)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * len(runs) + 2.0), 5.2))
    palette = [
        "#4472c4",
        "#ed7d31",
        "#a5a5a5",
        "#ffc000",
        "#5b9bd5",
        "#70ad47",
        "#264478",
        "#9e480e",
        "#636363",
        "#987300",
    ]
    for i, (k, vals) in enumerate(series):
        ax.bar(labels, vals, bottom=bottoms, label=k, color=palette[i % len(palette)])
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.bar(labels, other_ms, bottom=bottoms, label="(other)", color="#d9d9d9")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"CT×PT mul_nvtx breakdown (top {len(top_kernels)} kernels)")
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "tsv",
        nargs="+",
        type=Path,
        help="Input *_kern_sum.mul_nvtx_clean.tsv files (from export_ct_pt_kern_sum.py)",
    )
    ap.add_argument("--labels", nargs="*", default=None, help="Optional labels for each TSV (same count as inputs)")
    ap.add_argument("--top-n", type=int, default=6, help="Number of top kernels to show in breakdown")
    ap.add_argument("--baseline", type=str, default=None, help="Label to use as baseline for speedup reporting")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: directory of first TSV)")
    ap.add_argument("--tag", type=str, default="ct_pt_mul_nvtx_compare", help="Output filename tag")
    args = ap.parse_args()

    tsvs: List[Path] = [p for p in args.tsv]
    if args.labels is not None and len(args.labels) not in (0, len(tsvs)):
        raise SystemExit("--labels must be omitted or have the same count as input TSVs")

    out_dir = args.out_dir or tsvs[0].resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Run] = []
    for i, p in enumerate(tsvs):
        label = args.labels[i] if args.labels else _auto_label(p)
        runs.append(Run(label=label, path=p, totals_ns_by_kernel=_read_clean_tsv(p)))

    top_kernels = _pick_top_kernels(runs, args.top_n)
    out_total_png = out_dir / f"{args.tag}_total.png"
    out_break_png = out_dir / f"{args.tag}_breakdown.png"
    out_json = out_dir / f"{args.tag}_summary.json"

    # plots
    _plot_total_bar(out_total_png, runs)
    _plot_breakdown_stacked(out_break_png, runs, top_kernels)

    # summary
    baseline_label = args.baseline or (runs[0].label if runs else None)
    baseline_ns = None
    for r in runs:
        if r.label == baseline_label:
            baseline_ns = r.total_ns
            break
    if baseline_ns is None and runs:
        baseline_ns = runs[0].total_ns
        baseline_label = runs[0].label

    summary = {
        "baseline": baseline_label,
        "runs": [
            {
                "label": r.label,
                "path": str(r.path),
                "total_ns": r.total_ns,
                "total_ms": _ms(r.total_ns),
                "speedup_vs_baseline": (baseline_ns / r.total_ns) if (baseline_ns and r.total_ns) else None,
                "top_kernels_ns": {k: float(r.totals_ns_by_kernel.get(k, 0.0)) for k in top_kernels},
            }
            for r in runs
        ],
        "top_kernels": top_kernels,
        "artifacts": {
            "total_png": str(out_total_png),
            "breakdown_png": str(out_break_png),
            "summary_json": str(out_json),
        },
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {out_total_png}")
    print(f"Wrote: {out_break_png}")
    print(f"Wrote: {out_json}")
    if baseline_label:
        for r in runs:
            if baseline_ns and r.total_ns:
                print(f"speedup {baseline_label} -> {r.label}: {baseline_ns / r.total_ns:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

