#!/usr/bin/env python3
"""
Clean nsys `cuda_gpu_kern_sum` TSV (with optional preamble lines) and emit
spreadsheet-friendly TSV plus bar/pie charts.

Depends on matplotlib for PNGs; without it, only the clean TSV is written.

Usage:
  ./export_ct_pt_kern_sum.py path/to/kern_sum.mul_nvtx.tsv
      # writes *_clean.tsv and *_bar.png next to the input TSV (same directory)
  ./export_ct_pt_kern_sum.py --out-dir /tmp/out a.tsv b.tsv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent


def _load_parse_module():
    path = _SCRIPT_DIR / "parse_nsys_cuda_kern_sum.py"
    spec = importlib.util.spec_from_file_location("parse_nsys_cuda_kern_sum", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"export_ct_pt_kern_sum: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_p = _load_parse_module()
find_table_header = _p.find_table_header
col_idx = _p.col_idx


def short_kernel_name(full: str) -> str:
    full = full.strip()
    if "(" in full:
        return full.split("(", 1)[0].rstrip()
    return full or "unknown"


def read_kern_sum_rows(lines: list[str]) -> tuple[list[str], str, list[dict[str, str]]]:
    header, delim = find_table_header(lines)
    lowered = [h.strip().lower() for h in header]

    def pick(*needles: str) -> int | None:
        for needle in needles:
            for j, h in enumerate(lowered):
                if needle in h:
                    return j
        return None

    idx_pct = pick("time (%)", "time%")
    idx_total = col_idx(header, "total time")
    idx_inst = pick("instances")
    idx_avg = pick("avg")
    idx_name = col_idx(header, "name", "kernel")

    rows_out: list[dict[str, str]] = []
    for raw in lines:
        if not raw.strip() or raw.startswith("#"):
            continue
        fields = next(csv.reader([raw.rstrip("\n")], delimiter=delim))
        if fields == header:
            continue
        if len(fields) <= max(idx_total, idx_name):
            continue
        joined = " ".join(f.lower() for f in fields)
        if "total time" in joined and ("name" in joined or "kernel" in joined):
            continue
        try:
            float(fields[idx_total].replace(",", ""))
        except (ValueError, IndexError):
            continue
        row: dict[str, str] = {
            "Name": fields[idx_name] if idx_name < len(fields) else "",
            "ShortName": short_kernel_name(fields[idx_name] if idx_name < len(fields) else ""),
        }
        if idx_pct is not None and idx_pct < len(fields):
            row["Time (%)"] = fields[idx_pct].strip()
        row["Total Time (ns)"] = fields[idx_total].replace(",", "").strip()
        if idx_inst is not None and idx_inst < len(fields):
            row["Instances"] = fields[idx_inst].strip()
        if idx_avg is not None and idx_avg < len(fields):
            row["Avg (ns)"] = fields[idx_avg].strip()
        rows_out.append(row)

    # Stable sort: descending by total time
    def sort_key(r: dict[str, str]) -> float:
        try:
            return float(r["Total Time (ns)"])
        except ValueError:
            return 0.0

    rows_out.sort(key=sort_key, reverse=True)
    return header, delim, rows_out


def write_clean_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("# no data rows\n", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_charts(out_base: Path, rows: list[dict[str, str]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "export_ct_pt_kern_sum: matplotlib not installed; skipped PNG charts "
            "(pip install matplotlib)",
            file=sys.stderr,
        )
        return

    if not rows:
        return

    labels = [r["ShortName"] for r in rows]
    totals = [float(r["Total Time (ns)"]) for r in rows]
    pcts: list[float] = []
    for r in rows:
        try:
            pcts.append(float(r.get("Time (%)", "0").replace(",", "")))
        except ValueError:
            pcts.append(0.0)

    # Horizontal bar — total time
    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.35 * len(labels) + 1)))
    y_pos = range(len(labels))
    ax.barh(list(y_pos), totals, color="#4472c4")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Total Time (ns)")
    ax.set_title("CUDA kernel sum (by total time)")
    fig.tight_layout()
    fig.savefig(f"{out_base}_bar.png", dpi=150)
    plt.close(fig)

    # Pie — time % (only rows with positive pct)
    pie_labels: list[str] = []
    pie_vals: list[float] = []
    for r, p in zip(rows, pcts):
        if p > 0:
            pie_labels.append(r["ShortName"])
            pie_vals.append(p)
    if pie_vals and sum(pie_vals) > 0:
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.pie(pie_vals, labels=pie_labels, autopct="%1.1f%%", textprops={"fontsize": 8})
        ax2.set_title("Time (%) share")
        fig2.tight_layout()
        fig2.savefig(f"{out_base}_pie.png", dpi=150)
        plt.close(fig2)


def process_one(tsv_path: Path, out_dir: Path) -> None:
    text = tsv_path.read_text(errors="replace")
    lines = text.splitlines()
    _header, _delim, rows = read_kern_sum_rows(lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = tsv_path.stem  # e.g. ct_pt_micro_run1.kern_sum.mul_nvtx
    clean_path = out_dir / f"{stem}_clean.tsv"
    chart_base = out_dir / stem
    write_clean_tsv(clean_path, rows)
    write_charts(chart_base, rows)
    print(f"wrote {clean_path}")
    if rows:
        bar_p = Path(f"{chart_base}_bar.png")
        pie_p = Path(f"{chart_base}_pie.png")
        parts = [str(bar_p)]
        if pie_p.is_file():
            parts.append(str(pie_p))
        print("wrote " + " ; ".join(parts))
    else:
        print("no kernel rows; charts skipped", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean nsys kern_sum TSV and emit charts.")
    ap.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="kern_sum*.tsv files (cuda_gpu_kern_sum TSV)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for *_clean.tsv and *_bar.png (default: same folder as each input .tsv)",
    )
    args = ap.parse_args()

    for p in args.inputs:
        if not p.is_file():
            print(f"export_ct_pt_kern_sum: skip missing file {p}", file=sys.stderr)
            continue
        pr = p.resolve()
        out_dir = args.out_dir.resolve() if args.out_dir is not None else pr.parent
        process_one(pr, out_dir)


if __name__ == "__main__":
    main()
