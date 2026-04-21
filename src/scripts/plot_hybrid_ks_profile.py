#!/usr/bin/env python3
"""
Stacked bar chart from sim_hybrid_ks_profile CSV (Fig.1(b)-style workload breakdown).

Generate CSV (default: skip α with |Ql|≥α and |Ql| mod α≠0 — set MOAI_SIM_HYBRID_KS_EXACT_PARTITION=0 to keep all α):
  MOAI_TEST_OUTPUT_DISABLE=1 MOAI_BENCH_MODE=sim_hybrid_ks_profile \\
    MOAI_SIM_HYBRID_KS_ALPHA_RANGE=1-35 ./build/test

Engine schedule + coeff_ops:
  MOAI_SIM_BACKEND=1 MOAI_SIM_HYBRID_KS_MEASURE_ENGINE=1 ...

Stacked + engine compare + memory + **simulator cycles** + **donut %** (needs MEASURE_ENGINE CSV for cycle pie):
  python3 src/scripts/plot_hybrid_ks_profile.py --all-plots

Montgomery VEC plot (`--montgomery-cycles`): engine `vec_arith` estimate adds
`eng_vec_arith_mac_ops * mac_cyc` (default `mac_cyc` = `--mmul-cyc`).
BConv (`vec_bconv`) default: FMA-style `madd*mac_c + max(0,mmul-madd)*mmul_c` (matches EngineModel KS BConv FMA).
Use `--montgomery-bconv-split-mmadd` for legacy `mmul*mmul_c + madd*madd_c`.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _format_mod_sizes_table(
    *,
    alphas: list[int],
    size_Ql: list[int],
    t_qp: list[int],
    general: list[int | None],
    bts: int,
    special: list[int],
) -> str:
    headers = ["alpha", "size_Ql", "BTS", "GENERAL(=size_Ql-BTS)", "SPECIAL(=α)", "t_qp(=size_Ql+α)"]
    rows: list[list[str]] = []
    for a, ql, tq, g, s in zip(alphas, size_Ql, t_qp, general, special):
        g_str = str(g) if g is not None else "N/A"
        rows.append([str(a), str(ql), str(bts), g_str, str(s), str(tq)])
    widths = [len(h) for h in headers]
    for r in rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len(v))

    def fmt_row(cols: list[str]) -> str:
        return "  ".join(cols[i].rjust(widths[i]) for i in range(len(cols)))

    out = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    out.extend(fmt_row(r) for r in rows)
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("output/sim/hybrid_ks_profile.csv"),
        help="Path to hybrid_ks_profile.csv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("output/sim/hybrid_ks_profile_stacked.png"),
        help="Output PNG path (stacked bars)",
    )
    ap.add_argument(
        "--metric",
        choices=["kernels", "weighted"],
        default="kernels",
        help="kernels: ntt_fwd+ntt_inv+bconv+vec counts. weighted: weighted_ntt_fwd/inv + bconv + vec.",
    )
    ap.add_argument(
        "--yscale",
        choices=["raw", "1e8"],
        default="raw",
        help="For weighted only: divide Y by 1e8. Kernels mode uses raw counts.",
    )
    ap.add_argument(
        "--compare-engine",
        action="store_true",
        help="Second figure: ntt_fwd / ntt_inv / VEC BConv vs vec arith analytic vs engine enqueue_calls",
    )
    ap.add_argument(
        "--all-plots",
        action="store_true",
        help="Shorthand: write stacked PNG + engine compare PNG (same as --compare-engine)",
    )
    ap.add_argument(
        "--out-compare",
        type=Path,
        default=Path("output/sim/hybrid_ks_profile_engine_compare.png"),
        help="Output path for --compare-engine",
    )
    ap.add_argument(
        "--memory",
        action="store_true",
        help="Plot mem_* bytes vs α (Phantom t_mod_up + cx + c2; key_bytes_est line)",
    )
    ap.add_argument(
        "--out-memory",
        type=Path,
        default=Path("output/sim/hybrid_ks_profile_memory.png"),
        help="Output path for --memory",
    )
    ap.add_argument(
        "--cycles-plots",
        action="store_true",
        help="Y = EngineModel busy_cycles / makespan (not structural counts); needs eng_*_busy_cyc columns",
    )
    ap.add_argument(
        "--ntt-cycles",
        action="store_true",
        help="Plot NTT busy_cycles (total/fwd/inv) vs α; needs eng_ntt_*_busy_cyc columns",
    )
    ap.add_argument(
        "--out-cycles-stacked",
        type=Path,
        default=Path("output/sim/hybrid_ks_profile_stacked_cycles.png"),
        help="Stacked bars: ntt_fwd/inv + vec_bconv + vec_arith busy_cycles",
    )
    ap.add_argument(
        "--out-cycles-lines",
        type=Path,
        default=Path("output/sim/hybrid_ks_profile_engine_cycles.png"),
        help="Lines: makespan + per-engine busy_cycles vs alpha",
    )
    ap.add_argument(
        "--out-ntt-cycles",
        type=Path,
        default=Path("output/sim/hybrid_ks_profile_ntt_cycles.png"),
        help="Output path for --ntt-cycles",
    )
    ap.add_argument(
        "--pie",
        action="store_true",
        help="Donut chart(s): % share of workload (sum over CSV rows); optional engine busy split",
    )
    ap.add_argument(
        "--out-pie",
        type=Path,
        default=Path("output/sim/hybrid_ks_profile_pie.png"),
        help="Output path for --pie (structural + optional simulator busy_cycles)",
    )
    ap.add_argument(
        "--montgomery-cycles",
        action="store_true",
        help="Plot VEC-cycle estimate using Montgomery MMUL/MADD op counts from CSV (final reduce absorbed)",
    )
    ap.add_argument(
        "--mmul-cyc",
        type=float,
        default=1.0,
        help="Cycles per modular multiply (effective; include lanes/throughput in this number)",
    )
    ap.add_argument(
        "--madd-cyc",
        type=float,
        default=1.0,
        help="Cycles per modular add (effective; include lanes/throughput in this number)",
    )
    ap.add_argument(
        "--mac-cyc",
        type=float,
        default=None,
        help="Cycles per fused MAC op for eng_vec_arith_mac_ops (default: same as --mmul-cyc)",
    )
    ap.add_argument(
        "--montgomery-bconv-split-mmadd",
        action="store_true",
        help="BConv only: estimate cycles as MMUL+MADD separately. Default: FMA (MAC + extra MMUL when mmul>madd).",
    )
    ap.add_argument(
        "--out-montgomery-cycles",
        type=Path,
        default=Path("output/sim/hybrid_ks_profile_montgomery_cycles.png"),
        help="Output path for --montgomery-cycles",
    )
    ap.add_argument(
        "--mod-sizes",
        action="store_true",
        help="Plot modulus sizes vs α using fixed T and BTS: GENERAL=(T-BTS-α), BTS fixed, SPECIAL=α",
    )
    ap.add_argument(
        "--T",
        type=int,
        default=36,
        help="Total modulus size T = |QlP| = GENERAL + BTS + SPECIAL(=α)",
    )
    ap.add_argument(
        "--bts",
        type=int,
        default=14,
        help="Fixed BTS modulus size used for security (kept constant while sweeping α)",
    )
    ap.add_argument(
        "--out-mod-sizes",
        type=Path,
        default=Path("output/sim/hybrid_ks_profile_mod_sizes.png"),
        help="Output path for --mod-sizes",
    )
    ap.add_argument(
        "--mod-sizes-txt",
        action="store_true",
        help="Print/save α→(GENERAL,BTS,SPECIAL,T) as aligned text (more readable than a plot)",
    )
    ap.add_argument(
        "--out-mod-sizes-txt",
        type=Path,
        default=Path("output/sim/hybrid_ks_profile_mod_sizes.txt"),
        help="Output path for --mod-sizes-txt (table)",
    )
    ap.add_argument(
        "--mod-sizes-skip-na",
        action="store_true",
        help="When printing modulus size table, skip rows where GENERAL would be negative (size_Ql < BTS)",
    )
    args = ap.parse_args()
    if args.all_plots:
        args.compare_engine = True
        args.memory = True
        args.cycles_plots = True
        args.ntt_cycles = True
        args.pie = True
        args.mod_sizes = True
        args.mod_sizes_txt = True
        args.montgomery_cycles = True

    if not args.csv.is_file():
        raise SystemExit(f"missing {args.csv} (run sim_hybrid_ks_profile first)")

    rows: list[dict[str, str]] = []
    with args.csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    if not rows:
        raise SystemExit("empty csv")

    def alpha_key(row: dict[str, str]) -> float:
        return float(row["alpha"])

    rows.sort(key=alpha_key)

    alphas: list[float] = []
    bottom1: list[float] = []
    bottom2: list[float] = []
    bottom3: list[float] = []
    bottom4: list[float] = []
    bottom5: list[float] = []
    has_ntt_split = "ntt_fwd_kernels" in rows[0]

    for row in rows:
        a = float(row["alpha"])
        alphas.append(a)
        div = 1e8 if args.metric == "weighted" and args.yscale == "1e8" else 1.0
        if args.metric == "kernels":
            bconv = float(row["bconv_modup"]) + float(row["bconv_moddown"])
            vm = float(row["vec_mul"])
            va = float(row["vec_add"])
            if has_ntt_split:
                bottom1.append(float(row["ntt_fwd_kernels"]) / div)
                bottom2.append(float(row["ntt_inv_kernels"]) / div)
                bottom3.append(bconv / div)
                bottom4.append(vm / div)
                bottom5.append(va / div)
            else:
                bottom1.append(float(row["ntt_kernels"]) / div)
                bottom2.append(bconv / div)
                bottom3.append(vm / div)
                bottom4.append(va / div)
                bottom5.append(0.0)
        else:
            bconv = float(row["weighted_bconv_coeff_elems"])
            vm = float(row["weighted_vec_mul_coeff_elems"])
            va = float(row.get("weighted_vec_add_coeff_elems", "0") or 0)
            if has_ntt_split:
                bottom1.append(float(row["weighted_ntt_fwd_coeff_elems"]) / div)
                bottom2.append(float(row["weighted_ntt_inv_coeff_elems"]) / div)
                bottom3.append(bconv / div)
                bottom4.append(vm / div)
                bottom5.append(float(va) / div)
            else:
                bottom1.append(float(row["weighted_ntt_coeff_elems"]) / div)
                bottom2.append(bconv / div)
                bottom3.append(vm / div)
                bottom4.append(float(va) / div)
                bottom5.append(0.0)

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install with: pip install matplotlib\n" + str(e)
        ) from e

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(len(alphas)))
    width = 0.65
    suf = " (kernels)" if args.metric == "kernels" else " (weighted)"
    if has_ntt_split:
        ax.bar(x, bottom1, width, label="NTT fwd" + suf)
        b1 = bottom1
        ax.bar(x, bottom2, width, bottom=b1, label="INTT" + suf)
        b12 = [b1[i] + bottom2[i] for i in range(len(x))]
        ax.bar(x, bottom3, width, bottom=b12, label="BConv (up+down)" + suf)
        b123 = [b12[i] + bottom3[i] for i in range(len(x))]
        ax.bar(x, bottom4, width, bottom=b123, label="vec_mul" + suf)
        b1234 = [b123[i] + bottom4[i] for i in range(len(x))]
        ax.bar(x, bottom5, width, bottom=b1234, label="vec_add" + suf)
    else:
        ax.bar(x, bottom1, width, label="NTT (fwd+inv)" + suf)
        ax.bar(x, bottom2, width, bottom=bottom1, label="BConv (up+down)" + suf)
        bottom12 = [bottom1[i] + bottom2[i] for i in range(len(x))]
        ax.bar(x, bottom3, width, bottom=bottom12, label="vec_mul" + suf)
        bottom123 = [bottom12[i] + bottom3[i] for i in range(len(x))]
        ax.bar(x, bottom4, width, bottom=bottom123, label="vec_add" + suf)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(a)) if a == int(a) else str(a) for a in alphas])
    ax.set_xlabel(r"$\alpha$ (digit / special_modulus_size, sorted)")
    if args.metric == "kernels":
        ax.set_ylabel("Kernel / pass count (structural)")
    else:
        ylab = "Weighted coeff workload"
        if args.yscale == "1e8":
            ylab += r" ($\times 10^8$ coeff-elem proxy)"
        ax.set_ylabel(ylab)
    ax.set_title(f"Hybrid keyswitch breakdown — metric={args.metric}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    if args.ntt_cycles:
        need = ["eng_ntt_busy_cyc", "eng_ntt_fwd_busy_cyc", "eng_ntt_inv_busy_cyc"]
        missing = [c for c in need if c not in rows[0]]
        if missing:
            print(f"missing columns for --ntt-cycles: {missing} (re-run sim_hybrid_ks_profile with MEASURE_ENGINE enabled)")
        else:
            n_tot = [float(r.get("eng_ntt_busy_cyc", 0) or 0) for r in rows]
            n_fwd = [float(r.get("eng_ntt_fwd_busy_cyc", 0) or 0) for r in rows]
            n_inv = [float(r.get("eng_ntt_inv_busy_cyc", 0) or 0) for r in rows]

            fig_n, ax_n = plt.subplots(figsize=(9, 4.6))
            ax_n.plot(alphas, n_tot, "o-", label="NTT busy (total)")
            ax_n.plot(alphas, n_fwd, "s-", label="NTT fwd busy")
            ax_n.plot(alphas, n_inv, "^-", label="INTT busy")
            ax_n.set_xlabel(r"$\alpha$")
            ax_n.set_ylabel("cycles")
            ax_n.set_title("Hybrid KS — NTT busy_cycles vs α (engine model)")
            ax_n.grid(True, alpha=0.25)
            ax_n.legend(loc="best", fontsize=9)
            fig_n.tight_layout()
            args.out_ntt_cycles.parent.mkdir(parents=True, exist_ok=True)
            fig_n.savefig(args.out_ntt_cycles, dpi=150)
            print(f"wrote {args.out_ntt_cycles}")

    if args.mod_sizes or args.mod_sizes_txt:
        T = int(args.T)
        bts = int(args.bts)
        if T <= 0 or bts < 0 or bts > T:
            raise SystemExit(f"invalid sizes: T={T}, bts={bts}")

        aa_s = [int(a) if a == int(a) else None for a in alphas]
        if any(v is None for v in aa_s):
            raise SystemExit("--mod-sizes expects integer α in CSV (got non-integer alpha)")
        aa_i = [int(v) for v in aa_s if v is not None]
        ql_i = [int(float(r["size_Ql"])) for r in rows]
        tq_i = [int(float(r.get("t_qp", T))) for r in rows]
        if len(ql_i) != len(aa_i):
            raise SystemExit("internal: size_Ql length mismatch")

        # User's intended partition example: alpha=1, size_Ql=35, BTS=14 => GENERAL=21.
        general_opt: list[int | None] = []
        keep_mask: list[bool] = []
        for ql in ql_i:
            g = ql - bts
            if g < 0:
                general_opt.append(None)
                keep_mask.append(False)
            else:
                general_opt.append(g)
                keep_mask.append(True)

        if args.mod_sizes_skip_na:
            aa_i2: list[int] = []
            ql_i2: list[int] = []
            tq_i2: list[int] = []
            general_opt2: list[int | None] = []
            for a, ql, tq, g, keep in zip(aa_i, ql_i, tq_i, general_opt, keep_mask):
                if keep:
                    aa_i2.append(a)
                    ql_i2.append(ql)
                    tq_i2.append(tq)
                    general_opt2.append(g)
            aa_i, ql_i, tq_i, general_opt = aa_i2, ql_i2, tq_i2, general_opt2

        special = aa_i
        bts_list = [bts] * len(aa_i)

        if args.mod_sizes_txt:
            txt = _format_mod_sizes_table(
                alphas=aa_i,
                size_Ql=ql_i,
                t_qp=tq_i,
                general=general_opt,
                bts=bts,
                special=special,
            )
            print(txt, end="")
            args.out_mod_sizes_txt.parent.mkdir(parents=True, exist_ok=True)
            args.out_mod_sizes_txt.write_text(txt, encoding="utf-8")
            print(f"wrote {args.out_mod_sizes_txt}")

        if args.mod_sizes:
            if any(g is None for g in general_opt):
                print("note: some α have size_Ql < BTS => GENERAL undefined; use --mod-sizes-skip-na to drop them")
            fig_s, ax_s = plt.subplots(figsize=(9, 4.6))
            xs = list(range(len(aa_i)))
            w = 0.65
            gen_plot = [int(g) if g is not None else 0 for g in general_opt]
            ax_s.bar(xs, gen_plot, w, label="GENERAL (=size_Ql-BTS)")
            ax_s.bar(xs, bts_list, w, bottom=gen_plot, label=f"BTS (fixed={bts})")
            gb = [gen_plot[i] + bts_list[i] for i in range(len(xs))]
            ax_s.bar(xs, special, w, bottom=gb, label="SPECIAL (=α)")
            ax_s.set_xticks(xs)
            ax_s.set_xticklabels([str(a) for a in aa_i])
            ax_s.set_xlabel(r"$\alpha$")
            ax_s.set_ylabel("modulus count (limbs)")
            ax_s.set_title(f"Partition vs α (t_qp≈{T} fixed, BTS={bts} fixed)")
            ax_s.legend(loc="upper right", fontsize=8)
            fig_s.tight_layout()
            args.out_mod_sizes.parent.mkdir(parents=True, exist_ok=True)
            fig_s.savefig(args.out_mod_sizes, dpi=150)
            print(f"wrote {args.out_mod_sizes}")

    if args.pie:

        def _donut(ax, sizes: list[float], labels: list[str], title: str) -> None:
            pairs = [(s, lb) for s, lb in zip(sizes, labels) if s > 0]
            if not pairs:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                ax.set_axis_off()
                return
            sz, lb = zip(*pairs)
            ax.pie(
                sz,
                labels=lb,
                autopct="%1.1f%%",
                startangle=90,
                pctdistance=0.72,
                labeldistance=1.06,
                textprops={"fontsize": 9},
                wedgeprops=dict(width=0.45, edgecolor="white"),
            )
            ax.set_title(title, fontsize=10)

        s1, s2, s3, s4, s5 = (sum(bottom1), sum(bottom2), sum(bottom3), sum(bottom4), sum(bottom5))
        suf = " (kernels)" if args.metric == "kernels" else " (weighted)"
        if has_ntt_split:
            struct_labels = ["NTT fwd", "INTT", "BConv", "vec_mul", "vec_add"]
            struct_sizes = [s1, s2, s3, s4, s5]
        else:
            struct_labels = ["NTT (fwd+inv)", "BConv", "vec_mul", "vec_add"]
            struct_sizes = [s1, s2, s3, s4]
        struct_title = f"Structural share{suf}\n(sum over α in CSV)"

        has_cyc_pie = bool(
            rows
            and "eng_makespan_cyc" in rows[0]
            and any(int(float(r.get("eng_makespan_cyc", -1) or -1)) >= 0 for r in rows)
        )
        cyc_title = "Simulator busy_cycles share\n(sum over α; pipes overlap ⇒ ≠ wall time)"
        if has_cyc_pie:
            has_vec_cyc_split = "eng_vec_bconv_busy_cyc" in rows[0]
            nf_b = [float(r["eng_ntt_fwd_busy_cyc"]) for r in rows]
            ni_b = [float(r["eng_ntt_inv_busy_cyc"]) for r in rows]
            if has_vec_cyc_split:
                vbconv = [float(r["eng_vec_bconv_busy_cyc"]) for r in rows]
                varith = [float(r["eng_vec_arith_busy_cyc"]) for r in rows]
                cyc_labels = ["NTT fwd busy", "INTT busy", "VEC BConv busy", "VEC arith busy"]
                cyc_sizes = [sum(nf_b), sum(ni_b), sum(vbconv), sum(varith)]
            else:
                vb = [float(r["eng_vec_busy_cyc"]) for r in rows]
                cyc_labels = ["NTT fwd busy", "INTT busy", "VEC busy"]
                cyc_sizes = [sum(nf_b), sum(ni_b), sum(vb)]

        n_plots = 2 if has_cyc_pie else 1
        fig_p, axes = plt.subplots(1, n_plots, figsize=(6.3 * n_plots, 5.4))
        if n_plots == 1:
            axes = [axes]
        _donut(axes[0], struct_sizes, struct_labels, struct_title)
        if has_cyc_pie:
            _donut(axes[1], cyc_sizes, cyc_labels, cyc_title)
        fig_p.suptitle(f"Hybrid KS — metric={args.metric}", fontsize=11, y=1.03)
        fig_p.tight_layout()
        args.out_pie.parent.mkdir(parents=True, exist_ok=True)
        fig_p.savefig(args.out_pie, dpi=150, bbox_inches="tight")
        print(f"wrote {args.out_pie}")

    if args.montgomery_cycles:
        need_cols = [
            "bconv_mmul_ops",
            "bconv_madd_ops",
            "vec_mul_mmul_ops",
            "vec_add_madd_ops",
        ]
        missing = [c for c in need_cols if c not in rows[0]]
        if missing:
            print(f"missing columns for --montgomery-cycles: {missing} (re-run sim_hybrid_ks_profile with updated build)")
        else:
            aa = [float(r["alpha"]) for r in rows]
            mmul_c = float(args.mmul_cyc)
            madd_c = float(args.madd_cyc)
            mac_c = float(args.mac_cyc) if args.mac_cyc is not None else mmul_c
            bconv_fma = not bool(args.montgomery_bconv_split_mmadd)

            def cyc(mmul_ops: list[float], madd_ops: list[float]) -> list[float]:
                return [mmul_ops[i] * mmul_c + madd_ops[i] * madd_c for i in range(len(mmul_ops))]

            def bconv_cycle_est(mmul_ops: list[float], madd_ops: list[float]) -> list[float]:
                # Aligns with engine_model.h vec_bconv_montgomery_service_cycles (lane-free scalar proxy).
                if bconv_fma:
                    return [
                        madd_ops[i] * mac_c + max(0.0, mmul_ops[i] - madd_ops[i]) * mmul_c
                        for i in range(len(mmul_ops))
                    ]
                return cyc(mmul_ops, madd_ops)

            b_mmul_a = [float(r["bconv_mmul_ops"]) for r in rows]
            b_madd_a = [float(r["bconv_madd_ops"]) for r in rows]
            has_split = "bconv_modup_mmul_ops" in rows[0]
            if has_split:
                up_mmul_a = [float(r["bconv_modup_mmul_ops"]) for r in rows]
                up_madd_a = [float(r["bconv_modup_madd_ops"]) for r in rows]
                dn_mmul_a = [float(r["bconv_moddown_mmul_ops"]) for r in rows]
                dn_madd_a = [float(r["bconv_moddown_madd_ops"]) for r in rows]
                up_cyc_a = bconv_cycle_est(up_mmul_a, up_madd_a)
                dn_cyc_a = bconv_cycle_est(dn_mmul_a, dn_madd_a)
            else:
                up_cyc_a = []
                dn_cyc_a = []
            a_mmul_a = [float(r["vec_mul_mmul_ops"]) for r in rows]
            a_madd_a = [float(r["vec_add_madd_ops"]) for r in rows]
            bcyc_a = bconv_cycle_est(b_mmul_a, b_madd_a)
            acyc_a = cyc(a_mmul_a, a_madd_a)
            total_a = [bcyc_a[i] + acyc_a[i] for i in range(len(rows))]

            has_eng_ops = "eng_vec_bconv_mmul_ops" in rows[0] and any(
                int(float(r.get("eng_vec_bconv_mmul_ops", -1) or -1)) >= 0 for r in rows
            )
            has_eng_arith_mac = "eng_vec_arith_mac_ops" in rows[0]
            a_mac_e: list[float] = []
            if has_eng_ops:
                b_mmul_e = [float(r["eng_vec_bconv_mmul_ops"]) for r in rows]
                b_madd_e = [float(r["eng_vec_bconv_madd_ops"]) for r in rows]
                a_mmul_e = [float(r["eng_vec_arith_mmul_ops"]) for r in rows]
                a_madd_e = [float(r["eng_vec_arith_madd_ops"]) for r in rows]
                a_mac_e = [float(r.get("eng_vec_arith_mac_ops", 0) or 0) for r in rows] if has_eng_arith_mac else [0.0] * len(rows)
                bcyc_e = bconv_cycle_est(b_mmul_e, b_madd_e)
                acyc_e = [
                    a_mmul_e[i] * mmul_c + a_madd_e[i] * madd_c + a_mac_e[i] * mac_c for i in range(len(rows))
                ]
                total_e = [bcyc_e[i] + acyc_e[i] for i in range(len(rows))]
            else:
                bcyc_e = []
                acyc_e = []
                total_e = []

            bconv_lbl = "BConv FMA est (mac+extra×mmul)" if bconv_fma else "BConv MMUL+MADD est"
            fig_m, (axm1, axm2) = plt.subplots(1, 2, figsize=(12, 4.3))
            axm1.plot(aa, bcyc_a, "o-", label=f"analytic vec_bconv total ({bconv_lbl})")
            if has_split:
                axm1.plot(aa, up_cyc_a, "^-", alpha=0.85, label=f"analytic modup ({bconv_lbl})")
                axm1.plot(aa, dn_cyc_a, "v-", alpha=0.85, label=f"analytic moddown×2 ({bconv_lbl})")
            if has_eng_ops:
                axm1.plot(aa, bcyc_e, "s--", label=f"engine vec_bconv ({bconv_lbl})")
            axm1.set_xlabel(r"$\alpha$")
            axm1.set_ylabel("estimated cycles")
            axm1.set_title("BConv (vec_bconv)")
            axm1.legend(fontsize=8)

            axm2.plot(aa, total_a, "o-", label="analytic total VEC (bconv+arith; arith=mul+madd ops only)")
            axm2.plot(aa, acyc_a, "^-", alpha=0.9, label="analytic vec_arith (mul+madd)")
            if has_eng_ops:
                axm2.plot(aa, total_e, "s--", label="engine total VEC (bconv+arith incl. MAC ops)")
                lbl_arith = (
                    "engine vec_arith (mmul+madd+mac)"
                    if has_eng_arith_mac and a_mac_e and sum(a_mac_e) > 0
                    else "engine vec_arith (mmul+madd)"
                )
                axm2.plot(aa, acyc_e, "v--", alpha=0.9, label=lbl_arith)
            axm2.set_xlabel(r"$\alpha$")
            axm2.set_ylabel("estimated cycles")
            axm2.set_title("VEC total vs arith")
            axm2.legend(fontsize=8)

            bconv_mode = "BConv=FMA" if bconv_fma else "BConv=MMUL+MADD"
            fig_m.suptitle(
                f"Montgomery VEC estimate — {bconv_mode}; mmul={mmul_c:g}, madd={madd_c:g}, mac={mac_c:g} cyc/op",
                fontsize=11,
            )
            fig_m.tight_layout()
            args.out_montgomery_cycles.parent.mkdir(parents=True, exist_ok=True)
            fig_m.savefig(args.out_montgomery_cycles, dpi=150)
            print(f"wrote {args.out_montgomery_cycles}")

    if args.compare_engine and "eng_ntt_enq" in rows[0]:
        has_eng = any(int(float(r.get("eng_ntt_enq", -1) or -1)) >= 0 for r in rows)
        if not has_eng:
            print("no eng_* data (re-run with MOAI_SIM_BACKEND=1 MOAI_SIM_HYBRID_KS_MEASURE_ENGINE=1)")
        else:
            aa = [float(r["alpha"]) for r in rows]
            vec_a = [
                float(r["bconv_modup"]) + float(r["bconv_moddown"]) + float(r["vec_mul"]) + float(r["vec_add"])
                for r in rows
            ]
            vec_e = [float(r["eng_vec_enq"]) for r in rows]
            split_cmp = "eng_ntt_fwd_enq" in rows[0]
            vec_eng_split = split_cmp and "eng_vec_bconv_enq" in rows[0]
            if split_cmp:
                nf_a = [float(r["ntt_fwd_kernels"]) for r in rows]
                nf_e = [float(r["eng_ntt_fwd_enq"]) for r in rows]
                ni_a = [float(r["ntt_inv_kernels"]) for r in rows]
                ni_e = [float(r["eng_ntt_inv_enq"]) for r in rows]
                if vec_eng_split:
                    bconv_a = [float(r["bconv_modup"]) + float(r["bconv_moddown"]) for r in rows]
                    bconv_e = [float(r["eng_vec_bconv_enq"]) for r in rows]
                    arith_a = [float(r["vec_mul"]) + float(r["vec_add"]) for r in rows]
                    arith_e = [float(r["eng_vec_arith_enq"]) for r in rows]
                    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
                    ax1.plot(aa, nf_a, "o-", label="analytic ntt_fwd")
                    ax1.plot(aa, nf_e, "s--", label="engine ntt_fwd")
                    ax1.set_xlabel(r"$\alpha$")
                    ax1.set_ylabel("count")
                    ax1.legend(fontsize=8)
                    ax1.set_title("NTT forward")
                    ax2.plot(aa, ni_a, "o-", label="analytic ntt_inv (INTT)")
                    ax2.plot(aa, ni_e, "s--", label="engine ntt_inv")
                    ax2.set_xlabel(r"$\alpha$")
                    ax2.set_ylabel("count")
                    ax2.legend(fontsize=8)
                    ax2.set_title("INTT (inverse)")
                    ax3.plot(aa, bconv_a, "o-", label="analytic BConv (modup+moddown)")
                    ax3.plot(aa, bconv_e, "s--", label="engine vec_bconv")
                    ax3.set_xlabel(r"$\alpha$")
                    ax3.set_ylabel("count")
                    ax3.legend(fontsize=8)
                    ax3.set_title("VEC pipe — BConv (keyswitch modup/moddown)")
                    ax4.plot(aa, arith_a, "o-", label="analytic vec_mul+vec_add")
                    ax4.plot(aa, arith_e, "s--", label="engine vec_arith")
                    ax4.set_xlabel(r"$\alpha$")
                    ax4.set_ylabel("count")
                    ax4.legend(fontsize=8)
                    ax4.set_title("VEC pipe — mul/add (arith)")
                else:
                    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
                    ax1.plot(aa, nf_a, "o-", label="analytic ntt_fwd")
                    ax1.plot(aa, nf_e, "s--", label="engine ntt_fwd")
                    ax1.set_xlabel(r"$\alpha$")
                    ax1.set_ylabel("count")
                    ax1.legend(fontsize=8)
                    ax1.set_title("NTT forward")
                    ax2.plot(aa, ni_a, "o-", label="analytic ntt_inv (INTT)")
                    ax2.plot(aa, ni_e, "s--", label="engine ntt_inv")
                    ax2.set_xlabel(r"$\alpha$")
                    ax2.set_ylabel("count")
                    ax2.legend(fontsize=8)
                    ax2.set_title("INTT (inverse)")
                    ax3.plot(aa, vec_a, "o-", label="analytic vec stages")
                    ax3.plot(aa, vec_e, "s--", label="engine vec")
                    ax3.set_xlabel(r"$\alpha$")
                    ax3.set_ylabel("count")
                    ax3.legend(fontsize=8)
                    ax3.set_title("VEC (bconv+mul+add)")
            else:
                ntt_a = [float(r["ntt_kernels"]) for r in rows]
                ntt_e = [float(r["eng_ntt_enq"]) for r in rows]
                fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
                ax1.plot(aa, ntt_a, "o-", label="analytic ntt_kernels")
                ax1.plot(aa, ntt_e, "s--", label="engine ntt enqueue_calls")
                ax1.set_xlabel(r"$\alpha$")
                ax1.set_ylabel("count")
                ax1.legend()
                ax1.set_title("NTT")
                ax2.plot(aa, vec_a, "o-", label="analytic vec stages (bconv+mul+add)")
                ax2.plot(aa, vec_e, "s--", label="engine vec enqueue_calls")
                ax2.set_xlabel(r"$\alpha$")
                ax2.set_ylabel("count")
                ax2.legend()
                ax2.set_title("VEC engine (all vec schedules)")
            fig2.suptitle("Analytic vs EngineModel schedule() counts (keyswitch core only)")
            fig2.tight_layout()
            args.out_compare.parent.mkdir(parents=True, exist_ok=True)
            fig2.savefig(args.out_compare, dpi=150)
            print(f"wrote {args.out_compare}")

    if args.memory and rows and "mem_c2_bytes" in rows[0]:
        aa = [float(r["alpha"]) for r in rows]
        c2_b = [float(r["mem_c2_bytes"]) for r in rows]
        up_b = [float(r["mem_modup_buf_bytes"]) for r in rows]
        cx_b = [float(r["mem_cx_buf_bytes"]) for r in rows]
        peak_b = [float(r["mem_working_peak_bytes_est"]) for r in rows]
        key_b = [float(r.get("key_bytes_est", 0) or 0) for r in rows]
        gib = 1.0 / (1024.0**3)

        fig3, (axm1, axm2) = plt.subplots(1, 2, figsize=(11, 4))
        xb = list(range(len(aa)))
        w = 0.65
        c2_g = [v * gib for v in c2_b]
        up_g = [v * gib for v in up_b]
        cx_g = [v * gib for v in cx_b]
        axm1.bar(xb, c2_g, w, label="c2 (|Ql|)")
        axm1.bar(xb, up_g, w, bottom=c2_g, label=r"modup $t$ ($\beta\times$|QlP|)")
        b12 = [c2_g[i] + up_g[i] for i in range(len(xb))]
        axm1.bar(xb, cx_g, w, bottom=b12, label=r"cx (2$\times$|QlP|)")
        axm1.set_xticks(xb)
        axm1.set_xticklabels([str(int(a)) if a == int(a) else str(a) for a in aa])
        axm1.set_xlabel(r"$\alpha$")
        axm1.set_ylabel("Bytes (GiB)")
        axm1.set_title("KS working buffers (Phantom-style sizes)")
        axm1.legend(loc="upper right", fontsize=8)

        axm2.plot(aa, [p * gib for p in peak_b], "o-", label="mem_working_peak (c2+modup+cx)")
        axm2.plot(aa, [k * gib for k in key_b], "s--", label="key_bytes_est (relin key)")
        axm2.set_xlabel(r"$\alpha$")
        axm2.set_ylabel("Bytes (GiB)")
        axm2.set_title("Peak working vs relin key (est.)")
        axm2.legend(loc="best", fontsize=8)
        fig3.suptitle(
            "Hybrid keyswitch memory (u64 × count; see eval_key_switch.cu make_cuda_auto_ptr sizes)"
        )
        fig3.tight_layout()
        args.out_memory.parent.mkdir(parents=True, exist_ok=True)
        fig3.savefig(args.out_memory, dpi=150)
        print(f"wrote {args.out_memory}")
    elif args.memory:
        print("no mem_* columns in csv (re-run sim_hybrid_ks_profile with current MOAI_GPU test)")

    if args.cycles_plots and rows and "eng_makespan_cyc" in rows[0]:
        has_cyc = any(int(float(r.get("eng_makespan_cyc", -1) or -1)) >= 0 for r in rows)
        if not has_cyc:
            print("no cycle data (re-run with MOAI_SIM_BACKEND=1 MOAI_SIM_HYBRID_KS_MEASURE_ENGINE=1)")
        else:
            aa_c = [float(r["alpha"]) for r in rows]
            xc = list(range(len(aa_c)))
            w = 0.65
            nf_b = [float(r["eng_ntt_fwd_busy_cyc"]) for r in rows]
            ni_b = [float(r["eng_ntt_inv_busy_cyc"]) for r in rows]
            has_vec_cyc_split = "eng_vec_bconv_busy_cyc" in rows[0]
            vb = [float(r["eng_vec_busy_cyc"]) for r in rows]
            vbconv = [float(r["eng_vec_bconv_busy_cyc"]) for r in rows] if has_vec_cyc_split else vb
            varith = [float(r["eng_vec_arith_busy_cyc"]) for r in rows] if has_vec_cyc_split else [0.0] * len(rows)
            ms = [float(r["eng_makespan_cyc"]) for r in rows]
            ntt_tot = [float(r["eng_ntt_busy_cyc"]) for r in rows]

            fig_c1, ax_c1 = plt.subplots(figsize=(9, 5))
            ax_c1.bar(xc, nf_b, w, label="NTT fwd busy_cycles")
            ax_c1.bar(xc, ni_b, w, bottom=nf_b, label="INTT busy_cycles")
            b12c = [nf_b[i] + ni_b[i] for i in range(len(xc))]
            if has_vec_cyc_split:
                ax_c1.bar(xc, vbconv, w, bottom=b12c, label="VEC BConv busy_cycles")
                b123c = [b12c[i] + vbconv[i] for i in range(len(xc))]
                ax_c1.bar(
                    xc,
                    varith,
                    w,
                    bottom=b123c,
                    label="VEC arith busy_cycles (mul/add/MAC; non-BConv)",
                )
            else:
                ax_c1.bar(xc, vb, w, bottom=b12c, label="VEC busy_cycles (bconv+mul+add)")
            # Sanity: bconv + arith = eng_vec (arith includes enqueue_vec_mac).
            for i in range(len(rows)):
                ssplit = vbconv[i] + varith[i]
                if abs(ssplit - vb[i]) > 0.5:
                    print(
                        f"warning: row α={aa_c[i]} eng_vec_bconv+arith={ssplit:g} "
                        f"!= eng_vec_busy_cyc={vb[i]:g} (CSV column misalignment?)"
                    )
            ax_c1.set_xticks(xc)
            ax_c1.set_xticklabels([str(int(a)) if a == int(a) else str(a) for a in aa_c])
            ax_c1.set_xlabel(r"$\alpha$")
            ax_c1.set_ylabel("Simulated cycles (EngineStats.busy_cycles)")
            ax_c1.set_title(
                "Hybrid KS — per-engine busy (sum stacks work; overlap ⇒ ≠ wall time)\n"
                "VEC arith = all non–BConv vec (mul/add/MAC); BConv FMA/Montgomery stays in VEC BConv"
            )
            ax_c1.legend(loc="upper right", fontsize=8)
            fig_c1.tight_layout()
            args.out_cycles_stacked.parent.mkdir(parents=True, exist_ok=True)
            fig_c1.savefig(args.out_cycles_stacked, dpi=150)
            print(f"wrote {args.out_cycles_stacked}")

            fig_c2, ax_c2 = plt.subplots(figsize=(9, 4.5))
            ax_c2.plot(aa_c, ms, "k.-", linewidth=2, label="makespan_cycles (crit. path wall)")
            ax_c2.plot(aa_c, ntt_tot, "o-", label="ntt busy_cycles (fwd+inv, same pipe)")
            ax_c2.plot(aa_c, nf_b, "s--", alpha=0.85, label="ntt_fwd busy")
            ax_c2.plot(aa_c, ni_b, "^--", alpha=0.85, label="ntt_inv busy")
            if has_vec_cyc_split:
                ax_c2.plot(aa_c, vbconv, "D-", label="vec_bconv busy_cycles")
                ax_c2.plot(aa_c, varith, "v-", label="vec_arith (incl. MAC)")
                ax_c2.plot(aa_c, vb, ":", alpha=0.6, label="vec total (check)")
            else:
                ax_c2.plot(aa_c, vb, "D-", label="vec busy_cycles")
            ax_c2.set_xlabel(r"$\alpha$")
            ax_c2.set_ylabel("Cycles")
            ax_c2.set_title("EngineModel schedule — wall vs per-engine busy (keyswitch core)")
            ax_c2.legend(loc="best", fontsize=7)
            fig_c2.tight_layout()
            args.out_cycles_lines.parent.mkdir(parents=True, exist_ok=True)
            fig_c2.savefig(args.out_cycles_lines, dpi=150)
            print(f"wrote {args.out_cycles_lines}")
    elif args.cycles_plots:
        print("no eng_makespan_cyc in csv (rebuild test + MEASURE_ENGINE)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
