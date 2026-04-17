#!/usr/bin/env python3
"""
Plot CT×PT projection benchmark results produced by:
  bash src/scripts/bench_ct_pt_proj_compare.sh

Input:
  output/ct_pt_proj_compare/summary.json

Outputs (if data exists):
  - output/ct_pt_proj_compare/legacy_only.png
  - output/ct_pt_proj_compare/v2_only.png
  - output/ct_pt_proj_compare/compare.png   (only if both exist)
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["raw", "amortized_batch", "amortized_token", "amortized_batch_token"],
        default="amortized_batch",
        help="Y-axis normalization. Default: amortized per batch (divide by num_X).",
    )
    ap.add_argument(
        "--num-x",
        type=int,
        default=None,
        help="Override batch count (num_X). If omitted, uses summary.json meta.num_X when present.",
    )
    ap.add_argument(
        "--num-row",
        type=int,
        default=None,
        help="Override row count (num_row). If omitted, uses summary.json meta.num_row when present.",
    )
    args = ap.parse_args()

    root = Path.cwd().resolve()
    out = root / "output" / "ct_pt_proj_compare"
    summary_path = out / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"missing {summary_path} (run bench_ct_pt_proj_compare.sh first)")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = summary.get("runs", {})
    meta = summary.get("meta", {}) or {}
    # Backward-compatible defaults: older summary.json might not have meta.
    num_x = args.num_x if args.num_x is not None else int(meta.get("num_X") or 256)
    num_row = args.num_row if args.num_row is not None else int(meta.get("num_row") or 128)

    def norm(v: float) -> float:
        if args.mode == "raw":
            return v
        if args.mode == "amortized_batch":
            return v / num_x if num_x > 0 else v
        if args.mode == "amortized_token":
            return v / num_row if num_row > 0 else v
        if args.mode == "amortized_batch_token":
            denom = (num_x * num_row)
            return v / denom if denom > 0 else v
        return v

    def get(mode: str, op: str, key: str) -> float:
        return float(((runs.get(mode) or {}).get(op) or {}).get(key) or 0.0)

    labels = ["QKV_proj", "out_proj", "fc1", "fc2"]
    lv = [
        norm(get("legacy", "qkv", "QKV_proj")),
        norm(get("legacy", "out", "out_proj")),
        norm(get("legacy", "fc1", "fc1")),
        norm(get("legacy", "fc2", "fc2")),
    ]
    nv = [
        norm(get("v2", "qkv", "QKV_proj")),
        norm(get("v2", "out", "out_proj")),
        norm(get("v2", "fc1", "fc1")),
        norm(get("v2", "fc2", "fc2")),
    ]
    have_legacy = any(v > 0 for v in lv)
    have_v2 = any(v > 0 for v in nv)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"matplotlib unavailable: {e}")

    x = list(range(len(labels)))

    # Display in ms for readability (amortized times are usually small).
    def to_display_units(v: float) -> float:
        return v * 1e3

    def plot_one(path: Path, title: str, series: list[tuple[str, list[float], str]]):
        fig, ax = plt.subplots(figsize=(9.5, 4.4))
        if len(series) == 1:
            name, vals, color = series[0]
            bars = ax.bar(x, [to_display_units(v) for v in vals], 0.55, label=name, color=color)
        else:
            w = 0.38
            for idx, (name, vals, color) in enumerate(series):
                off = (-w / 2) if idx == 0 else (w / 2)
                bars = ax.bar([i + off for i in x], [to_display_units(v) for v in vals], w, label=name, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ylabel = "Time (ms)"
        if args.mode == "amortized_batch":
            ylabel = "Time (ms / batch)"
        elif args.mode == "amortized_token":
            ylabel = "Time (ms / token)"
        elif args.mode == "amortized_batch_token":
            ylabel = "Time (ms / batch·token)"
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False)
        # Annotate bar values with decimals.
        for rect in ax.patches:
            h = rect.get_height()
            if h <= 0:
                continue
            ax.annotate(f"{h:.3f}",
                        (rect.get_x() + rect.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8, rotation=0,
                        xytext=(0, 2), textcoords="offset points")
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)

    if have_legacy:
        plot_one(
            out / "legacy_only.png",
            f"CT×PT projection matmul — legacy ({args.mode})",
            [("legacy", lv, "#a5a5a5")],
        )
        print("Wrote", out / "legacy_only.png")
    if have_v2:
        plot_one(
            out / "v2_only.png",
            f"CT×PT projection matmul — v2 ({args.mode})",
            [("v2", nv, "#4472c4")],
        )
        print("Wrote", out / "v2_only.png")
    if have_legacy and have_v2:
        plot_one(
            out / "compare.png",
            f"CT×PT projection matmul — legacy vs v2 ({args.mode})",
            [("legacy", lv, "#a5a5a5"), ("v2", nv, "#4472c4")],
        )
        print("Wrote", out / "compare.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

