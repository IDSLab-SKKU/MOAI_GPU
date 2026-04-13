#!/usr/bin/env python3
"""
Parse `nsys stats --report cuda_gpu_kern_sum --format tsv` output.

Prints: total_time_ns, ntt_like_time_ns, ntt_fraction (of total), row count.
NTT-like rows match Phantom-style kernel names (heuristic regex).

Usage:
  nsys stats --report cuda_gpu_kern_sum --format tsv report.nsys-rep | ./parse_nsys_cuda_kern_sum.py
  ./parse_nsys_cuda_kern_sum.py kern_sum.tsv
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

NTT_LIKE = re.compile(
    r"(fnwt|inwt|nwt_2d|radix|intt|fntt|ntt_|inwt_|_nwt_)",
    re.IGNORECASE,
)


def sniff_delimiter(first_line: str) -> str:
    if "\t" in first_line:
        return "\t"
    return ","


def find_table_header(lines: list[str]) -> tuple[list[str], str]:
    for raw in lines:
        line = raw.strip("\n")
        if not line or line.startswith("#"):
            continue
        d = sniff_delimiter(line)
        fields = next(csv.reader([line], delimiter=d))
        joined = " ".join(fields).lower()
        if "total time" in joined and ("name" in joined or "kernel" in joined):
            return fields, d
    raise SystemExit("parse_nsys_cuda_kern_sum: could not find header row with Total Time and Name")


def col_idx(header: list[str], *needles: str) -> int:
    lowered = [h.strip().lower() for h in header]
    for needle in needles:
        for j, h in enumerate(lowered):
            if needle in h:
                return j
    raise SystemExit(f"parse_nsys_cuda_kern_sum: missing column matching {needles} in {header!r}")


def summarize(lines: list[str]) -> tuple[int, float, float, float]:
    header, delim = find_table_header(lines)
    idx_time = col_idx(header, "total time")
    idx_name = col_idx(header, "name", "kernel")

    total_ns = 0.0
    ntt_ns = 0.0
    rows = 0
    for raw in lines:
        if not raw.strip() or raw.startswith("#"):
            continue
        fields = next(csv.reader([raw.rstrip("\n")], delimiter=delim))
        if fields == header or len(fields) <= max(idx_time, idx_name):
            continue
        if "total time" in " ".join(f.lower() for f in fields):
            continue
        try:
            t = float(fields[idx_time].replace(",", ""))
        except ValueError:
            continue
        name = fields[idx_name] if idx_name < len(fields) else ""
        total_ns += t
        rows += 1
        if NTT_LIKE.search(name):
            ntt_ns += t

    frac = (ntt_ns / total_ns) if total_ns > 0 else 0.0
    return rows, total_ns, ntt_ns, frac


def main() -> None:
    if len(sys.argv) > 1:
        lines = Path(sys.argv[1]).read_text(errors="replace").splitlines()
    else:
        lines = sys.stdin.read().splitlines()

    rows, total_ns, ntt_ns, frac = summarize(lines)
    print(f"rows={rows}")
    print(f"total_time_ns={total_ns:.0f}")
    print(f"ntt_like_time_ns={ntt_ns:.0f}")
    print(f"ntt_like_fraction={frac:.6f}")


if __name__ == "__main__":
    main()
