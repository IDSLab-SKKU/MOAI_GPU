#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>

#include "source/sim/engine_config.h"
#include "source/sim/prng_evalkey_a.h"

namespace moai {
namespace sim {

// A coarse, dependency-aware engine scheduler:
// - Models 3 resources: DMA, NTT, VEC
// - Each enqueue returns a completion time in cycles
// - Engines overlap only when dependencies allow it
// This is intentionally rough; it is a stepping stone toward a full accelerator model.
class EngineModel {
 public:
  struct EngineStats {
    uint64_t busy_cycles = 0;
    uint64_t bytes = 0;
    // NTT/VEC: coefficient-elements (poly_degree * limbs scale) attributed per enqueue_* call.
    uint64_t logical_ops = 0;
    // Optional VEC op breakdown (Montgomery model; final reduction absorbed):
    uint64_t mmul_ops = 0;
    uint64_t madd_ops = 0;
    // Fused multiply-add (MAC / FMA) ops — `enqueue_vec_mac` coeff stream.
    uint64_t mac_ops = 0;
    uint64_t last_finish_cycles = 0;
    // schedule() invocations for this engine (DMA / NTT / vec / xfer buckets).
    uint64_t enqueue_calls = 0;
  };

  struct Summary {
    uint64_t makespan_cycles = 0;
    EngineStats dma{};
    EngineStats ntt{};
    // Same physical NTT pipe as `ntt`; split by enqueue_ntt_coeffs(..., ntt_inverse).
    EngineStats ntt_fwd{};
    EngineStats ntt_inv{};
    EngineStats vec{};
    // On-chip PRNG / random poly generation engine (e.g., SHAKE128+reject sampling).
    EngineStats prng{};
    // Same physical VEC pipe as `vec`; keyswitch modup/moddown `enqueue_vec_coeffs` vs mul/add.
    EngineStats vec_bconv{};
    // All non–keyswitch-BConv vec work: vec_mul, vec_add, enqueue_vec_mac (MAC ops in `mac_ops`).
    EngineStats vec_arith{};

    // On-chip transfer + capacity accounting (rough)
    EngineStats onchip_xfer{};
    EngineStats ntt_rf_xfer{};
    EngineStats vec_rf_xfer{};
    uint64_t gspad_bytes_used = 0;
    uint64_t gspad_spill_bytes = 0;
    uint64_t ntt_rf_spill_bytes = 0;
    uint64_t vec_rf_spill_bytes = 0;

    // Tiered byte accounting (logical accelerator view; ext = VRAM/HBM off-chip)
    uint64_t ext_read_bytes = 0;
    uint64_t ext_write_bytes = 0;
    uint64_t gspad_to_ntt_rf_bytes = 0;
    uint64_t ntt_rf_to_gspad_bytes = 0;
    uint64_t gspad_to_vec_rf_bytes = 0;
    uint64_t vec_rf_to_gspad_bytes = 0;
    uint64_t ntt_rf_to_vec_rf_bytes = 0;
    uint64_t vec_rf_to_ntt_rf_bytes = 0;
    uint64_t gspad_onchip_misc_bytes = 0;

    // Per-tier service cycles (sum of modeled xfer duration for bytes in each tier; overlaps across tiers possible).
    uint64_t ext_read_xfer_busy_cyc = 0;
    uint64_t ext_write_xfer_busy_cyc = 0;
    uint64_t gspad_to_ntt_rf_xfer_busy_cyc = 0;
    uint64_t ntt_rf_to_gspad_xfer_busy_cyc = 0;
    uint64_t gspad_to_vec_rf_xfer_busy_cyc = 0;
    uint64_t vec_rf_to_gspad_xfer_busy_cyc = 0;
    uint64_t ntt_rf_to_vec_rf_xfer_busy_cyc = 0;
    uint64_t vec_rf_to_ntt_rf_xfer_busy_cyc = 0;
    uint64_t gspad_onchip_misc_xfer_busy_cyc = 0;
  };

  static EngineModel &instance() {
    static EngineModel inst;
    return inst;
  }

  static bool enabled() {
    // Default: ON when MOAI_SIM_BACKEND=1, unless explicitly disabled.
    if (!SimBackendEnabled()) return false;
    if (const char *v = std::getenv("MOAI_SIM_ENGINE_MODEL"); v && v[0] != '\0')
      return std::strcmp(v, "0") != 0;
    return true;
  }

  void reset() {
    m_cfg = EngineModelConfig::from_env();
    m_onchip = OnChipConfig::from_env();
    m_dma = {};
    m_ntt = {};
    m_ntt_fwd = {};
    m_ntt_inv = {};
    m_vec = {};
    m_prng = {};
    m_vec_bconv = {};
    m_vec_arith = {};
    m_onchip_xfer = {};
    m_ntt_rf_xfer = {};
    m_vec_rf_xfer = {};
    m_gspad_used = 0;
    m_gspad_spill = 0;
    m_ntt_rf_spill = 0;
    m_vec_rf_spill = 0;
    m_pt_const_resident = false;
    m_pt_const_bytes = 0;
    m_makespan = 0;
    m_ext_read_bytes = 0;
    m_ext_write_bytes = 0;
    m_gspad_to_ntt_rf_bytes = 0;
    m_ntt_rf_to_gspad_bytes = 0;
    m_gspad_to_vec_rf_bytes = 0;
    m_vec_rf_to_gspad_bytes = 0;
    m_ntt_rf_to_vec_rf_bytes = 0;
    m_vec_rf_to_ntt_rf_bytes = 0;
    m_gspad_onchip_misc_bytes = 0;
    m_ext_read_xfer_busy_cyc = 0;
    m_ext_write_xfer_busy_cyc = 0;
    m_gspad_to_ntt_rf_xfer_busy_cyc = 0;
    m_ntt_rf_to_gspad_xfer_busy_cyc = 0;
    m_gspad_to_vec_rf_xfer_busy_cyc = 0;
    m_vec_rf_to_gspad_xfer_busy_cyc = 0;
    m_ntt_rf_to_vec_rf_xfer_busy_cyc = 0;
    m_vec_rf_to_ntt_rf_xfer_busy_cyc = 0;
    m_gspad_onchip_misc_xfer_busy_cyc = 0;
  }

  // Mark a plaintext constant as resident in GlobalSPAD (e.g., L(1) encode-once reuse).
  // If it doesn't fit, we keep it non-resident and treat as streaming.
  void mark_pt_const_resident(uint64_t bytes) {
    if (!m_onchip.enabled()) return;
    m_pt_const_bytes = bytes;
    m_pt_const_resident = ensure_gspad_capacity(bytes);
  }

  Summary summary() const {
    Summary s;
    s.makespan_cycles = m_makespan;
    s.dma = m_dma;
    s.ntt = m_ntt;
    s.ntt_fwd = m_ntt_fwd;
    s.ntt_inv = m_ntt_inv;
    s.vec = m_vec;
    s.prng = m_prng;
    s.vec_bconv = m_vec_bconv;
    s.vec_arith = m_vec_arith;
    s.onchip_xfer = m_onchip_xfer;
    s.ntt_rf_xfer = m_ntt_rf_xfer;
    s.vec_rf_xfer = m_vec_rf_xfer;
    s.gspad_bytes_used = m_gspad_used;
    s.gspad_spill_bytes = m_gspad_spill;
    s.ntt_rf_spill_bytes = m_ntt_rf_spill;
    s.vec_rf_spill_bytes = m_vec_rf_spill;
    s.ext_read_bytes = m_ext_read_bytes;
    s.ext_write_bytes = m_ext_write_bytes;
    s.gspad_to_ntt_rf_bytes = m_gspad_to_ntt_rf_bytes;
    s.ntt_rf_to_gspad_bytes = m_ntt_rf_to_gspad_bytes;
    s.gspad_to_vec_rf_bytes = m_gspad_to_vec_rf_bytes;
    s.vec_rf_to_gspad_bytes = m_vec_rf_to_gspad_bytes;
    s.ntt_rf_to_vec_rf_bytes = m_ntt_rf_to_vec_rf_bytes;
    s.vec_rf_to_ntt_rf_bytes = m_vec_rf_to_ntt_rf_bytes;
    s.gspad_onchip_misc_bytes = m_gspad_onchip_misc_bytes;
    s.ext_read_xfer_busy_cyc = m_ext_read_xfer_busy_cyc;
    s.ext_write_xfer_busy_cyc = m_ext_write_xfer_busy_cyc;
    s.gspad_to_ntt_rf_xfer_busy_cyc = m_gspad_to_ntt_rf_xfer_busy_cyc;
    s.ntt_rf_to_gspad_xfer_busy_cyc = m_ntt_rf_to_gspad_xfer_busy_cyc;
    s.gspad_to_vec_rf_xfer_busy_cyc = m_gspad_to_vec_rf_xfer_busy_cyc;
    s.vec_rf_to_gspad_xfer_busy_cyc = m_vec_rf_to_gspad_xfer_busy_cyc;
    s.ntt_rf_to_vec_rf_xfer_busy_cyc = m_ntt_rf_to_vec_rf_xfer_busy_cyc;
    s.vec_rf_to_ntt_rf_xfer_busy_cyc = m_vec_rf_to_ntt_rf_xfer_busy_cyc;
    s.gspad_onchip_misc_xfer_busy_cyc = m_gspad_onchip_misc_xfer_busy_cyc;
    return s;
  }

  void print_summary(std::ostream &os, const char *tag = "EngineModel") const {
    const std::ios::fmtflags saved_flags = os.flags();
    const Summary s = summary();
    const double period = (std::isfinite(m_cfg.sim_cycle_period_ns) && m_cfg.sim_cycle_period_ns > 0.0)
                              ? m_cfg.sim_cycle_period_ns
                              : 1.0;
    const double makespan_ns = static_cast<double>(s.makespan_cycles) * period;
    const double makespan_s = makespan_ns * 1e-9;
    const double eff_mhz = 1000.0 / period;

    os << std::fixed << std::setprecision(3);
    // makespan_* = simulated wall time for this schedule (critical path): all modeled DMA + on-chip +
    // NTT + VEC + RF traffic with overlap. Not the same as summing per-engine busy_cycles.
    os << "[MOAI_SIM_ENGINE] " << tag << " makespan_cycles=" << s.makespan_cycles
       << " makespan_s=" << makespan_s << " (crit.path wall: mem+engines, overlapped)\n";

    // Config sanity line (so logs are self-describing).
    const uint64_t gspad_eff_bw = std::max<uint64_t>(1, m_onchip.gspad_bw_gbps) * std::max<uint64_t>(1, m_onchip.gspad_banks);
    const uint64_t ntt_rf_eff_bw = std::max<uint64_t>(1, m_onchip.ntt_rf_bw_gbps) * std::max<uint64_t>(1, m_onchip.ntt_rf_banks);
    const uint64_t vec_rf_eff_bw = std::max<uint64_t>(1, m_onchip.vec_rf_bw_gbps) * std::max<uint64_t>(1, m_onchip.vec_rf_banks);
    os << "[MOAI_SIM_ENGINE] cfg"
       << " sim_cycle_period_ns=" << period
       << " (~" << eff_mhz << " MHz)"
       << " cycle_ns(int)=" << m_cfg.cycle_ns
       << " ext_mem=" << m_cfg.ext_mem.kind
       << " ext_bw_gbps=" << m_cfg.ext_mem.bandwidth_gbps
       << " ext_lat_ns=" << m_cfg.ext_mem.latency_ns
       << " ext_lat_eff_cyc=" << m_cfg.ext_mem.fixed_latency_cycles(m_cfg.sim_cycle_period_ns)
       << " gspad=" << fmt_bytes_iec(m_onchip.gspad_bytes) << " banks=" << m_onchip.gspad_banks
       << " bw_gbps=" << m_onchip.gspad_bw_gbps << " eff_bw_gbps=" << gspad_eff_bw
       << " ntt_rf=" << fmt_bytes_iec(m_onchip.ntt_rf_bytes) << " banks=" << m_onchip.ntt_rf_banks
       << " bw_gbps=" << m_onchip.ntt_rf_bw_gbps << " eff_bw_gbps=" << ntt_rf_eff_bw
       << " vec_rf=" << fmt_bytes_iec(m_onchip.vec_rf_bytes) << " banks=" << m_onchip.vec_rf_banks
       << " bw_gbps=" << m_onchip.vec_rf_bw_gbps << " eff_bw_gbps=" << vec_rf_eff_bw
       << " ntt_lanes=" << m_cfg.ntt_lanes
       << " ntt_pipe_depth_cyc=" << m_cfg.ntt_pipe_depth_cyc
       << " (auto_fill=log2(N) if depth=0)"
       << " ntt_steady_cyc_per_coeff=" << m_cfg.ntt_steady_cyc_per_coeff
       << " vec_lanes=" << m_cfg.vec_lanes
       << " vec_add_cyc=" << m_cfg.vec_add_cyc_per_coeff
       << " vec_mul_steady_cyc(db)=" << m_cfg.vec_mul_steady_cyc_per_coeff
       << " vec_mac_steady_cyc(db)=" << m_cfg.vec_mac_steady_cyc_per_coeff
       << " kswitch_P=" << m_cfg.kswitch_size_p << " ks_alpha=" << m_cfg.kswitch_digit_alpha
       << " ks_beta_mode=" << (m_cfg.kswitch_beta_legacy ? "legacy" : "phantom")
       << " kswitch_beta_env=" << m_cfg.kswitch_beta
       << " ks_modup_bconv_cyc=" << m_cfg.kswitch_modup_bconv_cyc_per_coeff
       << " ks_moddown_bconv_cyc=" << m_cfg.kswitch_moddown_bconv_cyc_per_coeff
       << " ks_bconv_montgomery_ops=" << (m_cfg.kswitch_bconv_use_montgomery_ops ? 1 : 0)
       << " ks_bconv_fma_ops=" << (m_cfg.kswitch_bconv_use_fma_ops ? 1 : 0)
       << " vec_mac_cyc_per_op=" << m_cfg.vec_mac_cyc_per_op
       << " galois_perm_cyc=" << m_cfg.galois_perm_cyc_per_coeff
       << " keyswitch_model=phantom_ckks_coarse"
       << " coeff_ops=NTT/VEC logical coeff-elements/op (op/s columns)"
       << "\n";

    const double ntt_steady_peak_gbps = theoretical_ntt_steady_coeff_equiv_gbps(m_cfg, period);
    const double vec_mul_peak_gbps = theoretical_vec_mul_steady_coeff_equiv_gbps(m_cfg, period);
    const double vec_add_peak_gbps = theoretical_vec_add_steady_coeff_equiv_gbps(m_cfg, period);
    const double vec_mac_peak_gbps = theoretical_vec_mac_steady_coeff_equiv_gbps(m_cfg, period);
    os << "[MOAI_SIM_ENGINE] steady_peak_coeff_equiv_GB/s (ideal lanes/steady, 8B/coeff; no fill/overhead): "
       << "ntt=" << fmt_gbps(ntt_steady_peak_gbps) << " vec_mul=" << fmt_gbps(vec_mul_peak_gbps)
       << " vec_add=" << fmt_gbps(vec_add_peak_gbps) << " vec_mac=" << fmt_gbps(vec_mac_peak_gbps)
       << " | onchip_RF_cfg_peak_GB/s ntt_rf_eff=" << static_cast<double>(ntt_rf_eff_bw)
       << " vec_rf_eff=" << static_cast<double>(vec_rf_eff_bw) << " (bw×banks)\n";

    auto row = [&](const char *name, const EngineStats &e) {
      const uint64_t gbps_bytes =
          (e.bytes != 0) ? e.bytes : static_cast<uint64_t>(e.logical_ops * 8ULL);
      const double gbps_busy = avg_throughput_gbps(gbps_bytes, e.busy_cycles, period);
      const double gbps_span = avg_throughput_gbps(gbps_bytes, s.makespan_cycles, period);
      const double opss_busy = avg_ops_per_sec(e.logical_ops, e.busy_cycles, period);
      const double opss_span = avg_ops_per_sec(e.logical_ops, s.makespan_cycles, period);
      os << "[MOAI_SIM_ENGINE] "
         << std::left << std::setw(12) << name
         << std::right << std::setw(14) << e.busy_cycles
         << std::setw(14) << fmt_bytes_iec(e.bytes)
         << std::setw(16) << e.last_finish_cycles
         << std::setw(12) << fmt_gbps(gbps_busy)
         << std::setw(12) << fmt_gbps(gbps_span)
         << std::setw(14) << e.logical_ops
         << std::setw(14) << fmt_ops_s(opss_busy)
         << std::setw(14) << fmt_ops_s(opss_span)
         << std::setw(10) << e.enqueue_calls
         << "\n";
    };

    os << "[MOAI_SIM_ENGINE] "
       << std::left << std::setw(12) << "engine"
       << std::right << std::setw(14) << "busy_cycles"
       << std::setw(14) << "bytes"
       << std::setw(16) << "last_finish"
       << std::setw(12) << "GB/s@busy"
       << std::setw(12) << "GB/s@span"
       << std::setw(14) << "coeff_ops"
       << std::setw(14) << "op/s@busy"
       << std::setw(14) << "op/s@span"
       << std::setw(10) << "enqueues"
       << "\n";
    os << "[MOAI_SIM_ENGINE] GB/s columns: byte engines use counted bytes; ntt/vec with bytes=0 use "
          "8B×coeff_ops (coeff-equiv) so GB/s aligns with op/s×8B.\n";
    row("dma", s.dma);
    row("ntt", s.ntt);
    row("ntt_fwd", s.ntt_fwd);
    row("ntt_inv", s.ntt_inv);
    row("vec", s.vec);
    row("prng", s.prng);
    row("vec_bconv", s.vec_bconv);
    row("vec_arith", s.vec_arith);
    row("onchip_xfer", s.onchip_xfer);
    row("ntt_rf_xfer", s.ntt_rf_xfer);
    row("vec_rf_xfer", s.vec_rf_xfer);

    // Coarse bound hint: critical tail (who finishes at makespan) + busy/makespan (can sum >1 with overlap).
    if (s.makespan_cycles > 0) {
      const uint64_t T = s.makespan_cycles;
      const char *names[] = {"dma", "onchip_xfer", "ntt_rf_xfer", "vec_rf_xfer", "ntt", "vec"};
      const uint64_t finishes[] = {s.dma.last_finish_cycles,
                                   s.onchip_xfer.last_finish_cycles,
                                   s.ntt_rf_xfer.last_finish_cycles,
                                   s.vec_rf_xfer.last_finish_cycles,
                                   s.ntt.last_finish_cycles,
                                   s.vec.last_finish_cycles};
      std::string crit;
      for (int i = 0; i < 6; ++i) {
        if (finishes[i] == T) {
          if (!crit.empty()) crit += '+';
          crit += names[i];
        }
      }
      if (crit.empty()) crit = "(none)";

      const double data_mov_busy_over_T =
          static_cast<double>(s.dma.busy_cycles + s.onchip_xfer.busy_cycles + s.ntt_rf_xfer.busy_cycles +
                              s.vec_rf_xfer.busy_cycles) /
          static_cast<double>(T);
      const double compute_busy_over_T =
          static_cast<double>(s.ntt.busy_cycles + s.vec.busy_cycles) / static_cast<double>(T);

      const bool tail_dma = (s.dma.last_finish_cycles == T);
      const bool tail_comp = (s.ntt.last_finish_cycles == T) || (s.vec.last_finish_cycles == T);
      const char *rough =
          (tail_dma && tail_comp) ? "mixed"
          : tail_dma               ? "device_mem_likely"
          : tail_comp              ? "compute_likely"
                                   : "onchip_move_likely";

      os << "[MOAI_SIM_ENGINE] bound_hint critical_tail=" << crit
         << " data_move_busy/makespan=" << std::setprecision(2) << data_mov_busy_over_T
         << " compute_busy/makespan=" << compute_busy_over_T << " rough=" << rough
         << " (Σbusy/makespan can exceed 1 when engines overlap)\n";

      // Engine "utilization" vs wall: busy_cycles / makespan (overlap ⇒ components can sum >100%).
      auto util_pct = [&](uint64_t busy) -> double {
        return 100.0 * static_cast<double>(busy) / static_cast<double>(T);
      };
      os << std::fixed << std::setprecision(1);
      os << "[MOAI_SIM_ENGINE] util_pct@ms"
         << " dma=" << util_pct(s.dma.busy_cycles) << "%"
         << " ntt=" << util_pct(s.ntt.busy_cycles) << "%"
         << " vec=" << util_pct(s.vec.busy_cycles) << "%"
         << " onchip=" << util_pct(s.onchip_xfer.busy_cycles) << "%"
         << " ntt_rf=" << util_pct(s.ntt_rf_xfer.busy_cycles) << "%"
         << " vec_rf=" << util_pct(s.vec_rf_xfer.busy_cycles) << "%\n";
      os << "[MOAI_SIM_ENGINE] util_agg@ms data_move%=" << (100.0 * data_mov_busy_over_T)
         << " compute%=" << (100.0 * compute_busy_over_T)
         << " (data_move=dma+onchip_xfer+ntt_rf_xfer+vec_rf; compute=ntt+vec; vs makespan wall)\n";
      os.unsetf(std::ios_base::floatfield);
      os << std::setprecision(3);
    }

    auto xfer_line = [&](const char *label, uint64_t raw_bytes, uint64_t tier_busy_cyc) {
      const double gbps_span = avg_throughput_gbps(raw_bytes, s.makespan_cycles, period);
      const double gbps_busy = avg_throughput_gbps(raw_bytes, tier_busy_cyc, period);
      os << "[MOAI_SIM_ENGINE] xfer\t" << std::left << std::setw(28) << label << std::right << std::setw(14)
         << fmt_bytes_iec(raw_bytes) << std::setw(16) << raw_bytes << std::setw(14) << tier_busy_cyc
         << std::setw(12) << fmt_gbps(gbps_span) << std::setw(12) << fmt_gbps(gbps_busy) << "\n";
    };
    os << "[MOAI_SIM_ENGINE] xfer_tier\tlabel\t\t\tbytes\tbytes_raw\ttier_busy_cyc\tavg_GB/s@span\tavg_GB/s@busy\n";
    xfer_line("ext_load (DMA<-ext)", s.ext_read_bytes, s.ext_read_xfer_busy_cyc);
    xfer_line("ext_store (DMA->ext)", s.ext_write_bytes, s.ext_write_xfer_busy_cyc);
    xfer_line("gspad<-ntt_rf", s.ntt_rf_to_gspad_bytes, s.ntt_rf_to_gspad_xfer_busy_cyc);
    xfer_line("gspad->ntt_rf", s.gspad_to_ntt_rf_bytes, s.gspad_to_ntt_rf_xfer_busy_cyc);
    xfer_line("gspad<-vec_rf", s.vec_rf_to_gspad_bytes, s.vec_rf_to_gspad_xfer_busy_cyc);
    xfer_line("gspad->vec_rf", s.gspad_to_vec_rf_bytes, s.gspad_to_vec_rf_xfer_busy_cyc);
    xfer_line("ntt_rf->vec_rf (peer)", s.ntt_rf_to_vec_rf_bytes, s.ntt_rf_to_vec_rf_xfer_busy_cyc);
    xfer_line("vec_rf->ntt_rf (peer)", s.vec_rf_to_ntt_rf_bytes, s.vec_rf_to_ntt_rf_xfer_busy_cyc);
    xfer_line("gspad_onchip_misc", s.gspad_onchip_misc_bytes, s.gspad_onchip_misc_xfer_busy_cyc);

    if (m_onchip.enabled())
    {
      os << "[MOAI_SIM_ENGINE] gspad_bytes_used=" << s.gspad_bytes_used
         << " gspad_spill_bytes=" << s.gspad_spill_bytes
         << " ntt_rf_spill_bytes=" << s.ntt_rf_spill_bytes
         << " vec_rf_spill_bytes=" << s.vec_rf_spill_bytes
         << "\n";
    }
    os.flags(saved_flags);
  }

  // -------------------------
  // Primitive enqueues
  // -------------------------
  // External memory (HBM/VRAM) via DMA — split read vs write for reporting.
  uint64_t enqueue_dma_read(uint64_t bytes, uint64_t dep_done_cycles = 0) {
    m_ext_read_bytes += bytes;
    const uint64_t service = dma_service_cycles(bytes);
    m_ext_read_xfer_busy_cyc += service;
    return schedule(m_dma, service, bytes, 0, dep_done_cycles);
  }

  uint64_t enqueue_dma_write(uint64_t bytes, uint64_t dep_done_cycles = 0) {
    m_ext_write_bytes += bytes;
    const uint64_t service = dma_service_cycles(bytes);
    m_ext_write_xfer_busy_cyc += service;
    return schedule(m_dma, service, bytes, 0, dep_done_cycles);
  }

  // Device↔device style: model as read then write at same byte volume.
  uint64_t enqueue_dma_d2d(uint64_t bytes, uint64_t dep_done_cycles = 0) {
    uint64_t t = enqueue_dma_read(bytes, dep_done_cycles);
    return enqueue_dma_write(bytes, t);
  }

  // Backward-compatible alias (treated as read — prefer enqueue_dma_read/write).
  uint64_t enqueue_dma(uint64_t bytes, uint64_t dep_done_cycles = 0) { return enqueue_dma_read(bytes, dep_done_cycles); }

  // Backward-compatible alias (older call sites).
  uint64_t enqueue_mem(uint64_t bytes, uint64_t dep_done_cycles = 0) { return enqueue_dma_read(bytes, dep_done_cycles); }

  enum class OnchipRoute : uint8_t {
    None,  // tier bytes already accounted (e.g. GSPAD↔RF via generic on-chip port)
    NttRfToGspad,
    VecRfToGspad,
    Misc,
  };

  uint64_t enqueue_onchip_xfer(uint64_t bytes, uint64_t dep_done_cycles = 0) {
    return enqueue_onchip_xfer(bytes, dep_done_cycles, OnchipRoute::Misc);
  }

  uint64_t enqueue_onchip_xfer(uint64_t bytes, uint64_t dep_done_cycles, OnchipRoute route) {
    const uint64_t service = onchip_service_cycles(bytes);
    switch (route) {
      case OnchipRoute::None:
        break;
      case OnchipRoute::NttRfToGspad:
        m_ntt_rf_to_gspad_bytes += bytes;
        m_ntt_rf_to_gspad_xfer_busy_cyc += service;
        break;
      case OnchipRoute::VecRfToGspad:
        m_vec_rf_to_gspad_bytes += bytes;
        m_vec_rf_to_gspad_xfer_busy_cyc += service;
        break;
      case OnchipRoute::Misc:
      default:
        m_gspad_onchip_misc_bytes += bytes;
        m_gspad_onchip_misc_xfer_busy_cyc += service;
        break;
    }
    return schedule(m_onchip_xfer, service, bytes, 0, dep_done_cycles);
  }

  uint64_t enqueue_ntt_rf_xfer(uint64_t bytes, uint64_t dep_done_cycles, bool from_gspad) {
    const uint64_t service =
        rf_service_cycles(bytes, m_onchip.ntt_rf_bw_gbps, m_onchip.ntt_rf_banks);
    if (from_gspad) {
      m_gspad_to_ntt_rf_bytes += bytes;
      m_gspad_to_ntt_rf_xfer_busy_cyc += service;
    } else {
      m_vec_rf_to_ntt_rf_bytes += bytes;
      m_vec_rf_to_ntt_rf_xfer_busy_cyc += service;
    }
    return schedule(m_ntt_rf_xfer, service, bytes, 0, dep_done_cycles);
  }

  uint64_t enqueue_vec_rf_xfer(uint64_t bytes, uint64_t dep_done_cycles, bool from_ntt_rf) {
    const uint64_t service =
        rf_service_cycles(bytes, m_onchip.vec_rf_bw_gbps, m_onchip.vec_rf_banks);
    if (from_ntt_rf) {
      m_ntt_rf_to_vec_rf_bytes += bytes;
      m_ntt_rf_to_vec_rf_xfer_busy_cyc += service;
    } else {
      m_gspad_to_vec_rf_bytes += bytes;
      m_gspad_to_vec_rf_xfer_busy_cyc += service;
    }
    return schedule(m_vec_rf_xfer, service, bytes, 0, dep_done_cycles);
  }

  // VEC pipe accounting: same `m_vec` schedule(), split into BConv-only vs all other vec (arith+MAC).
  enum class VecSplit : uint8_t {
    Arith,  // vec_mul, vec_add, enqueue_vec_mac, generic enqueue_vec_coeffs (non-keyswitch-bconv)
    BConv,  // keyswitch modup/moddown base conversion only
    Mac,    // enqueue_vec_mac — fused MAC; stats fold into `m_vec_arith` (mac_ops += coeffs)
  };

  void bump_vec_lane(VecSplit split,
                     uint64_t coeffs,
                     uint64_t service,
                     uint64_t finish,
                     uint64_t mmul_ops = 0,
                     uint64_t madd_ops = 0) {
    EngineStats &s = (split == VecSplit::BConv) ? m_vec_bconv : m_vec_arith;
    ++s.enqueue_calls;
    s.logical_ops += coeffs;
    if (split == VecSplit::Mac)
      s.mac_ops += coeffs;
    else {
      s.mmul_ops += mmul_ops;
      s.madd_ops += madd_ops;
    }
    s.busy_cycles += service;
    s.last_finish_cycles = finish;
  }

  // NTT: pipeline fill (see ntt_pipe_depth_cyc / auto log2(N)) + steady ceil(coeffs/lanes)*steady_cyc.
  // `ntt_inverse`: true = INTT (inverse) pass in this coarse model; false = forward NTT. Same `m_ntt` pipe for time.
  uint64_t enqueue_ntt_coeffs(uint64_t coeffs, uint64_t poly_degree, uint64_t dep_done_cycles = 0,
                              bool ntt_inverse = false) {
    const uint64_t service = ntt_service_cycles(coeffs, poly_degree);
    const uint64_t t_done = schedule(m_ntt, service, 0, coeffs, dep_done_cycles);
    EngineStats &split = ntt_inverse ? m_ntt_inv : m_ntt_fwd;
    ++split.enqueue_calls;
    split.logical_ops += coeffs;
    split.busy_cycles += service;
    split.last_finish_cycles = t_done;
    return t_done;
  }

  // PRNG / random poly generation: modeled as a coefficient stream engine (no bytes, only coeff_ops).
  // Intended for on-the-fly generation of keyswitch key component 'a' in NTT domain.
  uint64_t enqueue_prng_coeffs(uint64_t coeffs, uint64_t dep_done_cycles = 0) {
    const uint64_t service = prng_service_cycles(coeffs);
    return schedule(m_prng, service, 0, coeffs, dep_done_cycles);
  }

  // Generic vec op (e.g. rescale/modswitch); uses vec_lanes like mul/add.
  // `ks_keyswitch_bconv`: true only for Phantom CKKS keyswitch modup/moddown base-conv `enqueue_vec_coeffs` paths.
  uint64_t enqueue_vec_coeffs(uint64_t coeffs, uint64_t cycles_per_coeff, uint64_t dep_done_cycles = 0,
                              bool ks_keyswitch_bconv = false) {
    const uint64_t lanes = std::max<uint64_t>(1u, m_cfg.vec_lanes);
    const uint64_t waves = (coeffs + lanes - 1) / lanes;
    const uint64_t service = m_cfg.vec_pass_overhead_cyc + waves * cycles_per_coeff;
    const uint64_t t_done = schedule(m_vec, service, 0, coeffs, dep_done_cycles);
    bump_vec_lane(ks_keyswitch_bconv ? VecSplit::BConv : VecSplit::Arith, coeffs, service, t_done);
    return t_done;
  }

  uint64_t vec_montgomery_ops_service_cycles(uint64_t mmul_ops, uint64_t madd_ops) const {
    const uint64_t lanes = std::max<uint64_t>(1u, m_cfg.vec_lanes);
    const uint64_t mmul_waves = (mmul_ops + lanes - 1) / lanes;
    const uint64_t madd_waves = (madd_ops + lanes - 1) / lanes;
    const uint64_t mmul_cyc = std::max<uint64_t>(1u, m_cfg.vec_mmul_cyc_per_op);
    const uint64_t madd_cyc = std::max<uint64_t>(1u, m_cfg.vec_madd_cyc_per_op);
    return m_cfg.vec_pass_overhead_cyc + mmul_waves * mmul_cyc + madd_waves * madd_cyc;
  }

  // Keyswitch BConv Montgomery: separate MMUL+MADD waves, or FMA/MAC + optional extra MMUL-only ops.
  // FMA path: mac_ops = madd_ops (inner-product MACs); when mmul_ops > madd_ops, excess is phase1 mul-only
  // (moddown: mmul−madd = N·|P|). Modup has mmul==madd ⇒ all MAC.
  uint64_t vec_bconv_montgomery_service_cycles(uint64_t mmul_ops, uint64_t madd_ops) const {
    if (!m_cfg.kswitch_bconv_use_fma_ops)
      return vec_montgomery_ops_service_cycles(mmul_ops, madd_ops);
    const uint64_t lanes = std::max<uint64_t>(1u, m_cfg.vec_lanes);
    const uint64_t mac_cyc = std::max<uint64_t>(1u, m_cfg.vec_mac_cyc_per_op);
    const uint64_t mul_cyc = std::max<uint64_t>(1u, m_cfg.vec_mmul_cyc_per_op);
    const uint64_t extra_mul = (mmul_ops > madd_ops) ? (mmul_ops - madd_ops) : 0ULL;
    const uint64_t mac_ops = madd_ops;
    const uint64_t mac_waves = (mac_ops + lanes - 1) / lanes;
    const uint64_t extra_waves = (extra_mul + lanes - 1) / lanes;
    return m_cfg.vec_pass_overhead_cyc + mac_waves * mac_cyc + extra_waves * mul_cyc;
  }

  // Keyswitch BConv modeled as Montgomery MMUL/MADD matmul (final reduction absorbed).
  uint64_t enqueue_vec_bconv_montgomery(uint64_t coeffs,
                                        uint64_t mmul_ops,
                                        uint64_t madd_ops,
                                        uint64_t dep_done_cycles = 0) {
    const uint64_t service = vec_bconv_montgomery_service_cycles(mmul_ops, madd_ops);
    const uint64_t t_done = schedule(m_vec, service, 0, coeffs, dep_done_cycles);
    bump_vec_lane(VecSplit::BConv, coeffs, service, t_done, mmul_ops, madd_ops);
    return t_done;
  }

  uint64_t vec_mul_service_cycles(uint64_t coeffs) const {
    const uint64_t lanes = std::max<uint64_t>(1u, m_cfg.vec_lanes);
    const uint64_t waves = (coeffs + lanes - 1) / lanes;
    // Double-buffered MUL: pipe fill overlapped; charge only steady waves + pass overhead.
    return m_cfg.vec_pass_overhead_cyc + waves * m_cfg.vec_mul_steady_cyc_per_coeff;
  }

  uint64_t enqueue_vec_mul(uint64_t coeffs, uint64_t dep_done_cycles = 0) {
    const uint64_t service = vec_mul_service_cycles(coeffs);
    const uint64_t t_done = schedule(m_vec, service, 0, coeffs, dep_done_cycles);
    bump_vec_lane(VecSplit::Arith, coeffs, service, t_done, /*mmul_ops=*/coeffs, /*madd_ops=*/0);
    return t_done;
  }

  // Fused multiply-add on RNS coeff streams (same wave model as vec_mul; `mac_ops` in vec_arith).
  uint64_t vec_mac_service_cycles(uint64_t coeffs) const {
    const uint64_t lanes = std::max<uint64_t>(1u, m_cfg.vec_lanes);
    const uint64_t waves = (coeffs + lanes - 1) / lanes;
    return m_cfg.vec_pass_overhead_cyc + waves * m_cfg.vec_mac_steady_cyc_per_coeff;
  }

  uint64_t enqueue_vec_mac(uint64_t coeffs, uint64_t dep_done_cycles = 0) {
    const uint64_t service = vec_mac_service_cycles(coeffs);
    const uint64_t t_done = schedule(m_vec, service, 0, coeffs, dep_done_cycles);
    bump_vec_lane(VecSplit::Mac, coeffs, service, t_done);
    return t_done;
  }

  uint64_t enqueue_vec_add(uint64_t coeffs, uint64_t dep_done_cycles = 0) {
    const uint64_t t_done = enqueue_vec_coeffs(coeffs, m_cfg.vec_add_cyc_per_coeff, dep_done_cycles);
    // enqueue_vec_coeffs bumps vec_arith.logical_ops; add MADD accounting here.
    m_vec_arith.madd_ops += coeffs;
    return t_done;
  }

  // ct×pt multiply_plain breakdown: DMA read -> NTT fwd -> VEC mul -> NTT inv -> DMA write.
  uint64_t enqueue_multiply_plain(uint64_t ct_size,
                                 uint64_t poly_degree,
                                 uint64_t limbs,
                                 uint64_t dep_done_cycles = 0) {
    const uint64_t coeffs = poly_degree * limbs;
    const uint64_t bytes_ct = ct_size * coeffs * sizeof(uint64_t);
    const uint64_t bytes_pt = 1ULL * coeffs * sizeof(uint64_t);

    uint64_t t = dep_done_cycles;
    if (m_onchip.enabled())
    {
      // Chunk the whole op by NttRF capacity (RF = engine-local scratchpad).
      const uint64_t per_coeff_ct_bytes = ct_size * sizeof(uint64_t);
      const uint64_t chunk_coeffs =
          (m_onchip.ntt_rf_bytes != 0)
              ? std::max<uint64_t>(1, m_onchip.ntt_rf_bytes / per_coeff_ct_bytes)
              : coeffs;

      uint64_t done = 0;
      while (done < coeffs)
      {
        const uint64_t c = std::min<uint64_t>(chunk_coeffs, coeffs - done);
        const uint64_t bytes_ct_chunk = ct_size * c * sizeof(uint64_t);
        const uint64_t bytes_pt_chunk = 1ULL * c * sizeof(uint64_t);

        // DMA to GlobalSPAD (stage inputs). PT may be reused from resident constant.
        const uint64_t pt_dma = (m_pt_const_resident ? 0ULL : bytes_pt_chunk);
        t = enqueue_dma_read(bytes_ct_chunk + pt_dma, t);
        (void)ensure_gspad_capacity(bytes_ct_chunk + pt_dma);

        // PT: coefficient domain -> needs NTT fwd (GSPAD -> NttRF -> NTT -> NttRF -> VecRF).
        // CT: already in NTT/evaluation domain -> GSPAD -> VecRF only (no CT load into NttRF for fwd).
        const uint64_t pt_chunk_bytes = bytes_pt_chunk;
        if (!m_pt_const_resident && pt_chunk_bytes != 0)
        {
          t = stage_into_ntt_rf(pt_chunk_bytes, t, /*from_gspad=*/true);
          t = enqueue_ntt_coeffs(c, poly_degree, t, /*ntt_inverse=*/false);
          t = stage_into_vec_rf(pt_chunk_bytes, t, /*from_ntt_rf=*/true);
        }
        else if (m_pt_const_resident && pt_chunk_bytes != 0)
        {
          // Resident PT in SPAD; assumed already in eval (NTT) domain — skip PT NTT, stage to VecRF.
          t = stage_into_vec_rf(pt_chunk_bytes, t, /*from_ntt_rf=*/false);
        }

        t = stage_into_vec_rf(bytes_ct_chunk, t, /*from_gspad=*/false);
        t = enqueue_vec_mul(c, t);

        t = stage_into_ntt_rf(bytes_ct_chunk, t, /*from_gspad=*/false);
        t = enqueue_ntt_coeffs(c, poly_degree, t, /*ntt_inverse=*/true);

        // RF -> GlobalSPAD -> DMA writeback (chunk)
        t = enqueue_onchip_xfer(bytes_ct_chunk, t, OnchipRoute::NttRfToGspad);
        t = enqueue_dma_write(bytes_ct_chunk, t);

        done += c;
      }
    }
    else
    {
      t = enqueue_dma_read(bytes_ct + bytes_pt, t);
      t = enqueue_ntt_coeffs(coeffs, poly_degree, t, false);       // fwd
      t = enqueue_vec_mul(coeffs, t);                              // mul
      t = enqueue_ntt_coeffs(coeffs, poly_degree, t, true);       // inv
      t = enqueue_dma_write(bytes_ct, t);                         // write back ct
    }
    return t;
  }

  // CT×CT multiply (v0 coarse): two CTs from HBM, eval-domain staging, K×vec_mul (tensor/RNS proxy),
  // then NTT inv + writeback. Relinearize is modeled only when Evaluator calls enqueue_relinearize separately.
  uint64_t enqueue_ct_ct_multiply(uint64_t ct_size,
                                  uint64_t poly_degree,
                                  uint64_t limbs,
                                  uint64_t dep_done_cycles = 0) {
    const uint64_t coeffs = poly_degree * limbs;
    const uint64_t bytes_ct = ct_size * coeffs * sizeof(uint64_t);
    const uint64_t K = std::max<uint64_t>(1ULL, m_cfg.ct_ct_vec_mul_passes);

    uint64_t t = dep_done_cycles;
    if (m_onchip.enabled())
    {
      const uint64_t per_coeff_ct_bytes = ct_size * sizeof(uint64_t);
      const uint64_t chunk_coeffs =
          (m_onchip.ntt_rf_bytes != 0)
              ? std::max<uint64_t>(1, m_onchip.ntt_rf_bytes / per_coeff_ct_bytes)
              : coeffs;

      uint64_t done = 0;
      while (done < coeffs)
      {
        const uint64_t c = std::min<uint64_t>(chunk_coeffs, coeffs - done);
        const uint64_t bytes_ct_chunk = ct_size * c * sizeof(uint64_t);

        t = enqueue_dma_read(2ULL * bytes_ct_chunk, t);
        (void)ensure_gspad_capacity(2ULL * bytes_ct_chunk);

        // Both inputs in eval domain: contiguous GSPAD -> VecRF (v0 treats as one staging step).
        t = stage_into_vec_rf(2ULL * bytes_ct_chunk, t, /*from_ntt_rf=*/false);

        for (uint64_t k = 0; k < K; ++k)
          t = enqueue_vec_mul(c, t);

        t = stage_into_ntt_rf(bytes_ct_chunk, t, /*from_gspad=*/false);
        t = enqueue_ntt_coeffs(c, poly_degree, t, true);

        t = enqueue_onchip_xfer(bytes_ct_chunk, t, OnchipRoute::NttRfToGspad);
        t = enqueue_dma_write(bytes_ct_chunk, t);

        done += c;
      }
    }
    else
    {
      t = enqueue_dma_read(2ULL * bytes_ct, t);
      t = enqueue_ntt_coeffs(coeffs, poly_degree, t, false);
      for (uint64_t k = 0; k < K; ++k)
        t = enqueue_vec_mul(coeffs, t);
      t = enqueue_ntt_coeffs(coeffs, poly_degree, t, true);
      t = enqueue_dma_write(bytes_ct, t);
    }
    return t;
  }

  uint64_t enqueue_add_inplace(uint64_t ct_size,
                              uint64_t poly_degree,
                              uint64_t limbs,
                              uint64_t dep_done_cycles = 0) {
    const uint64_t coeffs = poly_degree * limbs;
    const uint64_t bytes_ct = ct_size * coeffs * sizeof(uint64_t);
    uint64_t t = dep_done_cycles;
    t = enqueue_dma_read(bytes_ct + bytes_ct, t);              // read acc + tmp
    t = enqueue_vec_add(coeffs, t);                              // add
    t = enqueue_dma_write(bytes_ct, t);                        // write acc
    return t;
  }

  uint64_t enqueue_rescale(uint64_t ct_size,
                           uint64_t poly_degree,
                           uint64_t limbs,
                           uint64_t dep_done_cycles = 0) {
    const uint64_t coeffs = poly_degree * limbs;
    const uint64_t bytes_ct = ct_size * coeffs * sizeof(uint64_t);
    uint64_t t = dep_done_cycles;
    t = enqueue_vec_coeffs(coeffs, m_cfg.rescale_cyc_per_coeff, t);
    if (m_onchip.enabled())
      t = enqueue_onchip_xfer(bytes_ct, t, OnchipRoute::VecRfToGspad);
    t = enqueue_dma_write(bytes_ct, t);
    return t;
  }

  // Galois rotate + keyswitch (Phantom evaluate.cu apply_galois_inplace CKKS path, then keyswitch_inplace).
  uint64_t enqueue_rotate(uint64_t ct_size, uint64_t poly_degree, uint64_t limbs, uint64_t dep_done_cycles = 0) {
    const uint64_t coeffs = poly_degree * limbs;
    const uint64_t bytes_poly = coeffs * sizeof(uint64_t);
    const uint64_t bytes_ct = ct_size * coeffs * sizeof(uint64_t);
    const uint64_t key_bytes = m_cfg.galois_key_bytes;
    uint64_t t = dep_done_cycles;
    const uint64_t key_dma_bytes =
        (m_cfg.otf_keygen ? static_cast<uint64_t>(std::llround(static_cast<double>(key_bytes) * m_cfg.otf_keygen_key_bytes_scale))
                          : key_bytes);
    const uint64_t t_dma = enqueue_dma_read(bytes_ct + key_dma_bytes, t);
    uint64_t t_prng = 0;
    if (m_cfg.otf_keygen) {
      // Model generating 'a' (random poly) on-chip in NTT domain. 1st-order: one poly on Ql.
      t_prng = enqueue_prng_coeffs(coeffs, t);
    }
    t = std::max(t_dma, t_prng);
    if (m_onchip.enabled())
      t = enqueue_onchip_xfer(bytes_ct + key_dma_bytes, t, OnchipRoute::Misc);
    // apply_galois_ntt on c0 then c1 + D2D staging (two polys × Ql)
    if (m_onchip.enabled())
      t = enqueue_onchip_xfer(2 * bytes_poly, t, OnchipRoute::Misc);
    t = enqueue_vec_coeffs(coeffs, m_cfg.galois_perm_cyc_per_coeff, t);
    t = enqueue_vec_coeffs(coeffs, m_cfg.galois_perm_cyc_per_coeff, t);
    t = enqueue_keyswitch_phantom_ckks(poly_degree, limbs, t);
    t = enqueue_dma_write(bytes_ct, t);
    return t;
  }

  // Relinearize (Phantom evaluate.cu relinearize_inplace -> keyswitch_inplace on c2, size 3 -> 2).
  uint64_t enqueue_relinearize(uint64_t ct_size, uint64_t poly_degree, uint64_t limbs, uint64_t dep_done_cycles = 0) {
    const uint64_t coeffs = poly_degree * limbs;
    const uint64_t bytes_poly = coeffs * sizeof(uint64_t);
    const uint64_t key_bytes = m_cfg.relin_key_bytes;
    (void)ct_size;  // API parity; relin always uses 3->2 polys at this level.
    uint64_t t = dep_done_cycles;
    const uint64_t bytes_in = 3 * bytes_poly;
    const uint64_t key_dma_bytes =
        (m_cfg.otf_keygen ? static_cast<uint64_t>(std::llround(static_cast<double>(key_bytes) * m_cfg.otf_keygen_key_bytes_scale))
                          : key_bytes);
    const uint64_t t_dma = enqueue_dma_read(bytes_in + key_dma_bytes, t);
    uint64_t t_prng = 0;
    if (m_cfg.otf_keygen) {
      t_prng = enqueue_prng_coeffs(coeffs, t);
    }
    t = std::max(t_dma, t_prng);
    if (m_onchip.enabled())
      t = enqueue_onchip_xfer(bytes_in + key_dma_bytes, t, OnchipRoute::Misc);
    t = enqueue_keyswitch_phantom_ckks(poly_degree, limbs, t);
    t = enqueue_dma_write(2 * bytes_poly, t);
    return t;
  }

  uint64_t enqueue_modswitch(uint64_t ct_size, uint64_t poly_degree, uint64_t limbs, uint64_t dep_done_cycles = 0) {
    const uint64_t coeffs = poly_degree * limbs;
    const uint64_t bytes_ct = ct_size * coeffs * sizeof(uint64_t);
    uint64_t t = dep_done_cycles;
    t = enqueue_vec_coeffs(coeffs, m_cfg.modswitch_cyc_per_coeff, t);
    if (m_onchip.enabled())
      t = enqueue_onchip_xfer(bytes_ct, t, OnchipRoute::VecRfToGspad);
    t = enqueue_dma_write(bytes_ct, t);
    return t;
  }

  // Hybrid KS op profiler: Phantom keyswitch core only (no DMA / Galois). Uses current env (MOAI_SIM_ALPHA, …).
  uint64_t profile_keyswitch_phantom_ckks(uint64_t poly_degree, uint64_t limbs, uint64_t dep_done_cycles = 0) {
    return enqueue_keyswitch_phantom_ckks(poly_degree, limbs, dep_done_cycles);
  }

 private:
  EngineModel() = default;

  static bool SimBackendEnabled() {
    if (const char *v = std::getenv("MOAI_SIM_BACKEND"); v && v[0] != '\0')
      return std::strcmp(v, "0") != 0;
    return false;
  }

  static uint64_t env_u64(const char *name, uint64_t def) {
    const char *v = std::getenv(name);
    if (!v || v[0] == '\0') return def;
    char *end = nullptr;
    unsigned long long x = std::strtoull(v, &end, 10);
    if (end == v) return def;
    return static_cast<uint64_t>(x);
  }

  uint64_t ntt_pipe_fill_cycles(uint64_t poly_degree) const {
    if (m_cfg.ntt_pipe_depth_cyc > 0) return m_cfg.ntt_pipe_depth_cyc;
    if (poly_degree >= 2) return floor_log2_u64(poly_degree);
    return 0;
  }

  uint64_t prng_service_cycles(uint64_t coeffs) const {
    const uint64_t lanes = std::max<uint64_t>(1u, m_cfg.prng_lanes);
    const uint64_t waves = (coeffs + lanes - 1) / lanes;
    const uint64_t cyc = std::max<uint64_t>(1u, m_cfg.prng_cyc_per_coeff);
    return m_cfg.prng_pass_overhead_cyc + waves * cyc;
  }

  uint64_t ntt_service_cycles(uint64_t coeffs, uint64_t poly_degree) const {
    const uint64_t lanes = std::max<uint64_t>(1u, m_cfg.ntt_lanes);
    const uint64_t fill = ntt_pipe_fill_cycles(poly_degree);
    const uint64_t waves = (coeffs + lanes - 1) / lanes;
    return m_cfg.ntt_pass_overhead_cyc + fill + waves * m_cfg.ntt_steady_cyc_per_coeff;
  }

  uint64_t dma_service_cycles(uint64_t bytes) const {
    // Per-transaction fixed latency (external memory / HBM row miss, MMU, DMA setup, …)
    const uint64_t latency = m_cfg.ext_mem.fixed_latency_cycles(m_cfg.sim_cycle_period_ns);
    // Streaming: xfer_ns ≈ ceil(bytes / BW) with BW in GB/s (ExternalMemoryConfig::bandwidth_gbps).
    const uint64_t bw = m_cfg.ext_mem.bandwidth_gbps;
    if (bw == 0) return latency;
    const uint64_t xfer_ns = (bytes + bw - 1) / bw;  // rough
    const double per = (std::isfinite(m_cfg.sim_cycle_period_ns) && m_cfg.sim_cycle_period_ns > 0.0)
                           ? m_cfg.sim_cycle_period_ns
                           : 1.0;
    const double xfer_cyc_d = std::ceil(static_cast<double>(xfer_ns) / per);
    const uint64_t xfer_cyc = static_cast<uint64_t>(std::max(1.0, xfer_cyc_d));
    return latency + xfer_cyc;
  }

  uint64_t onchip_service_cycles(uint64_t bytes) const {
    if (!m_onchip.enabled()) return 0;
    const uint64_t banks = std::max<uint64_t>(1, m_onchip.gspad_banks);
    const uint64_t bw = std::max<uint64_t>(1, m_onchip.gspad_bw_gbps * banks);
    const uint64_t xfer_ns = (bytes + bw - 1) / bw;
    const double per = (std::isfinite(m_cfg.sim_cycle_period_ns) && m_cfg.sim_cycle_period_ns > 0.0)
                           ? m_cfg.sim_cycle_period_ns
                           : 1.0;
    const double xfer_cyc_d = std::ceil(static_cast<double>(xfer_ns) / per);
    const uint64_t xfer_cyc = static_cast<uint64_t>(std::max(1.0, xfer_cyc_d));
    return m_onchip.gspad_base_cyc + xfer_cyc;
  }

  uint64_t rf_service_cycles(uint64_t bytes, uint64_t rf_bw_gbps, uint64_t rf_banks) const {
    if (!m_onchip.enabled()) return 0;
    const uint64_t banks = std::max<uint64_t>(1, rf_banks);
    const uint64_t bw = std::max<uint64_t>(1, rf_bw_gbps * banks);
    const uint64_t xfer_ns = (bytes + bw - 1) / bw;
    const double per = (std::isfinite(m_cfg.sim_cycle_period_ns) && m_cfg.sim_cycle_period_ns > 0.0)
                           ? m_cfg.sim_cycle_period_ns
                           : 1.0;
    const double xfer_cyc_d = std::ceil(static_cast<double>(xfer_ns) / per);
    const uint64_t xfer_cyc = static_cast<uint64_t>(std::max(1.0, xfer_cyc_d));
    return 1 + xfer_cyc; // small base for local staging
  }

  // Digit count beta for RNS keyswitch (Phantom DRNSTool::modup uses v_base_part_Ql_to_compl_part_QlP_conv size).
  // Phantom rns.cu: beta = ceil(|Ql|/alpha). Legacy: ceil(|Ql|/kswitch_size_p). MOAI_SIM_KSWITCH_BETA overrides.
  uint64_t keyswitch_beta_choose(uint64_t limbs) const {
    if (m_cfg.kswitch_beta > 0) return std::max<uint64_t>(1ULL, m_cfg.kswitch_beta);
    if (m_cfg.kswitch_beta_legacy) {
      const uint64_t P = std::max<uint64_t>(1ULL, m_cfg.kswitch_size_p);
      return std::max<uint64_t>(1ULL, (limbs + P - 1) / P);
    }
    const uint64_t alpha = std::max<uint64_t>(1ULL, m_cfg.kswitch_digit_alpha);
    return std::max<uint64_t>(1ULL, (limbs + alpha - 1) / alpha);
  }

  // CKKS keyswitch core: INTT Ql (c2) -> modup (bconv + NTT QlP) × beta -> inner prod × beta ->
  // moddown (INTT P-only + bconv + NTT Ql-only + fuse) × 2 -> add_to_ct × 2.
  // Note: Phantom modup forward-NTT excludes the digit's part-Ql range because it was already NTT in input.
  uint64_t enqueue_keyswitch_phantom_ckks(uint64_t poly_degree, uint64_t limbs, uint64_t t) {
    const uint64_t coeffs_ql = poly_degree * limbs;
    const uint64_t num_P = std::max<uint64_t>(1ULL, m_cfg.kswitch_size_p);
    const uint64_t size_QlP = limbs + num_P;
    const uint64_t coeffs_qlp = poly_degree * size_QlP;
    const uint64_t beta = keyswitch_beta_choose(limbs);
    const uint64_t alpha = std::max<uint64_t>(1ULL, m_cfg.kswitch_digit_alpha);
    const uint64_t coeffs_p = poly_degree * num_P;

    // On-the-fly eval-key 'a' generation request can start early (overlap with modup/bconv/ntt).
    // This models generating a directly in NTT domain and feeding it into the evk multiply stage.
    if (m_cfg.otf_keygen) {
      PrngTimingConfig pcfg;
      pcfg.num_prng_lanes = m_cfg.otf_evk_a_num_prng_lanes;
      pcfg.num_sampler_lanes = m_cfg.otf_evk_a_num_sampler_lanes;
      pcfg.shake_startup_cycles = m_cfg.otf_evk_a_shake_startup_cycles;
      pcfg.shake_block_cycles = m_cfg.otf_evk_a_shake_block_cycles;
      pcfg.bit_fifo_capacity_blocks = m_cfg.otf_evk_a_bit_fifo_capacity_blocks;
      pcfg.coeff_fifo_capacity = m_cfg.otf_evk_a_coeff_fifo_capacity;
      pcfg.chunk_size = m_cfg.otf_evk_a_chunk_size;
      pcfg.seed_metadata_bytes = m_cfg.otf_evk_a_seed_metadata_bytes;
      m_otf_evk_a.reset(pcfg);

      // Request for the evk-multiply stage: beta * QlP coeff stream + 2 * Ql stream (moddown mul).
      const uint64_t need = beta * coeffs_qlp + 2ULL * coeffs_ql;
      (void)m_otf_evk_a.request(t, need);
    }

    // modup: INTT on c2 (CKKS alpha==1 path in phantom-fhe/src/rns_bconv.cu DRNSTool::modup)
    t = enqueue_ntt_coeffs(coeffs_ql, poly_degree, t, true);

    if (m_onchip.enabled() && coeffs_qlp != 0 && beta != 0) {
      const uint64_t scratch = coeffs_qlp * sizeof(uint64_t) * beta;
      t = enqueue_onchip_xfer(scratch, t, OnchipRoute::Misc);
    }

    for (uint64_t bi = 0; bi < beta; ++bi) {
      const uint64_t start = alpha * bi;
      const uint64_t part_limbs = (start < limbs) ? std::min<uint64_t>(alpha, limbs - start) : 0ULL;
      const uint64_t coeffs_excl_range = poly_degree * (size_QlP - part_limbs);
      // Phantom modup BConv matmul ops:
      // out limbs = QlP \ partQl, in limbs = partQl.
      const uint64_t obase_size = size_QlP - part_limbs;
      const uint64_t ibase_size = part_limbs;
      const uint64_t mmul = poly_degree * obase_size * ibase_size;
      const uint64_t madd = poly_degree * obase_size * ibase_size;
      if (m_cfg.kswitch_bconv_use_montgomery_ops) {
        t = enqueue_vec_bconv_montgomery(coeffs_qlp, mmul, madd, t);
      } else {
        // Legacy coarse: constant cycles per coeff-element; still record ops for reporting.
        t = enqueue_vec_coeffs(coeffs_qlp, m_cfg.kswitch_modup_bconv_cyc_per_coeff, t, true);
        m_vec_bconv.mmul_ops += mmul;
        m_vec_bconv.madd_ops += madd;
      }
      // Phantom: forward NTT on QlP excluding [startPartIdx, endPartIdx) in Ql.
      t = enqueue_ntt_coeffs(coeffs_excl_range, poly_degree, t, false);
    }

    for (uint64_t bi = 0; bi < beta; ++bi) {
      (void)bi;
      if (m_cfg.otf_keygen) {
        // Gate evk multiply by a availability (consume QlP coeff stream).
        t = m_otf_evk_a.consume(t, coeffs_qlp);
      }
      t = enqueue_vec_mul(coeffs_qlp, t);
    }

    for (int part = 0; part < 2; ++part) {
      (void)part;
      // Phantom CKKS moddown_from_NTT: inverse-transform P only (Ql remains in NTT).
      t = enqueue_ntt_coeffs(coeffs_p, poly_degree, t, true);
      // Phantom moddown BConv ops (P -> Ql):
      // - phase1 mult over P limbs: N*alpha MMUL
      // - phase2 matmul: N*|Ql|*alpha MMUL + MADD
      const uint64_t ibase_size = num_P;
      const uint64_t obase_size = limbs;
      const uint64_t mmul = poly_degree * (ibase_size + obase_size * ibase_size);
      const uint64_t madd = poly_degree * (obase_size * ibase_size);
      if (m_cfg.kswitch_bconv_use_montgomery_ops) {
        t = enqueue_vec_bconv_montgomery(coeffs_ql, mmul, madd, t);
      } else {
        t = enqueue_vec_coeffs(coeffs_ql, m_cfg.kswitch_moddown_bconv_cyc_per_coeff, t, true);
        m_vec_bconv.mmul_ops += mmul;
        m_vec_bconv.madd_ops += madd;
      }
      // Phantom CKKS: nwt_2d_radix8_forward_inplace_fuse_moddown runs over Ql.
      t = enqueue_ntt_coeffs(coeffs_ql, poly_degree, t, false);
      if (m_cfg.otf_keygen) {
        // Moddown multiply also consumes 'a' over Ql (rough).
        t = m_otf_evk_a.consume(t, coeffs_ql);
      }
      t = enqueue_vec_mul(coeffs_ql, t);
    }

    t = enqueue_vec_add(coeffs_ql, t);
    t = enqueue_vec_add(coeffs_ql, t);
    return t;
  }

  uint64_t stage_into_ntt_rf(uint64_t bytes, uint64_t dep_done_cycles, bool from_gspad) {
    if (m_onchip.ntt_rf_bytes == 0) {
      const uint64_t svc = onchip_service_cycles(bytes);
      if (from_gspad) {
        m_gspad_to_ntt_rf_bytes += bytes;
        m_gspad_to_ntt_rf_xfer_busy_cyc += svc;
      } else {
        m_vec_rf_to_ntt_rf_bytes += bytes;
        m_vec_rf_to_ntt_rf_xfer_busy_cyc += svc;
      }
      return enqueue_onchip_xfer(bytes, dep_done_cycles, OnchipRoute::None);
    }
    if (bytes <= m_onchip.ntt_rf_bytes) {
      return enqueue_ntt_rf_xfer(bytes, dep_done_cycles, from_gspad);
    }
    // Stream in chunks; record spill as the bytes beyond capacity.
    m_ntt_rf_spill += (bytes - m_onchip.ntt_rf_bytes);
    uint64_t t = dep_done_cycles;
    uint64_t remaining = bytes;
    while (remaining) {
      const uint64_t chunk = std::min<uint64_t>(remaining, m_onchip.ntt_rf_bytes);
      t = enqueue_ntt_rf_xfer(chunk, t, from_gspad);
      remaining -= chunk;
    }
    return t;
  }

  uint64_t stage_into_vec_rf(uint64_t bytes, uint64_t dep_done_cycles, bool from_ntt_rf) {
    if (m_onchip.vec_rf_bytes == 0) {
      const uint64_t svc = onchip_service_cycles(bytes);
      if (from_ntt_rf) {
        m_ntt_rf_to_vec_rf_bytes += bytes;
        m_ntt_rf_to_vec_rf_xfer_busy_cyc += svc;
      } else {
        m_gspad_to_vec_rf_bytes += bytes;
        m_gspad_to_vec_rf_xfer_busy_cyc += svc;
      }
      return enqueue_onchip_xfer(bytes, dep_done_cycles, OnchipRoute::None);
    }
    if (bytes <= m_onchip.vec_rf_bytes) {
      return enqueue_vec_rf_xfer(bytes, dep_done_cycles, from_ntt_rf);
    }
    m_vec_rf_spill += (bytes - m_onchip.vec_rf_bytes);
    uint64_t t = dep_done_cycles;
    uint64_t remaining = bytes;
    while (remaining) {
      const uint64_t chunk = std::min<uint64_t>(remaining, m_onchip.vec_rf_bytes);
      t = enqueue_vec_rf_xfer(chunk, t, from_ntt_rf);
      remaining -= chunk;
    }
    return t;
  }

  bool ensure_gspad_capacity(uint64_t bytes_needed) {
    if (!m_onchip.enabled()) return true;
    if (bytes_needed > m_onchip.gspad_bytes)
    {
      m_gspad_spill += bytes_needed;
      return false;
    }
    if (m_gspad_used + bytes_needed > m_onchip.gspad_bytes)
    {
      // simple spill model: spill the incoming bytes
      m_gspad_spill += bytes_needed;
      return false;
    }
    m_gspad_used += bytes_needed;
    return true;
  }

  uint64_t schedule(EngineStats &e,
                     uint64_t service_cycles,
                     uint64_t bytes,
                     uint64_t logical_ops,
                     uint64_t dep_done_cycles) {
    ++e.enqueue_calls;
    const uint64_t start = std::max(e.last_finish_cycles, dep_done_cycles);
    const uint64_t finish = start + service_cycles;
    e.last_finish_cycles = finish;
    e.busy_cycles += service_cycles;
    e.bytes += bytes;
    e.logical_ops += logical_ops;
    m_makespan = std::max(m_makespan, finish);
    return finish;
  }

  EngineStats m_dma{};
  EngineStats m_ntt{};
  EngineStats m_ntt_fwd{};
  EngineStats m_ntt_inv{};
  EngineStats m_vec{};
  EngineStats m_prng{};
  PrngEvalKeyAGenerator m_otf_evk_a{};
  EngineStats m_vec_bconv{};
  EngineStats m_vec_arith{};
  EngineStats m_onchip_xfer{};
  EngineStats m_ntt_rf_xfer{};
  EngineStats m_vec_rf_xfer{};
  uint64_t m_gspad_used = 0;
  uint64_t m_gspad_spill = 0;
  uint64_t m_ntt_rf_spill = 0;
  uint64_t m_vec_rf_spill = 0;
  bool m_pt_const_resident = false;
  uint64_t m_pt_const_bytes = 0;
  uint64_t m_makespan = 0;

  uint64_t m_ext_read_bytes = 0;
  uint64_t m_ext_write_bytes = 0;
  uint64_t m_gspad_to_ntt_rf_bytes = 0;
  uint64_t m_ntt_rf_to_gspad_bytes = 0;
  uint64_t m_gspad_to_vec_rf_bytes = 0;
  uint64_t m_vec_rf_to_gspad_bytes = 0;
  uint64_t m_ntt_rf_to_vec_rf_bytes = 0;
  uint64_t m_vec_rf_to_ntt_rf_bytes = 0;
  uint64_t m_gspad_onchip_misc_bytes = 0;

  uint64_t m_ext_read_xfer_busy_cyc = 0;
  uint64_t m_ext_write_xfer_busy_cyc = 0;
  uint64_t m_gspad_to_ntt_rf_xfer_busy_cyc = 0;
  uint64_t m_ntt_rf_to_gspad_xfer_busy_cyc = 0;
  uint64_t m_gspad_to_vec_rf_xfer_busy_cyc = 0;
  uint64_t m_vec_rf_to_gspad_xfer_busy_cyc = 0;
  uint64_t m_ntt_rf_to_vec_rf_xfer_busy_cyc = 0;
  uint64_t m_vec_rf_to_ntt_rf_xfer_busy_cyc = 0;
  uint64_t m_gspad_onchip_misc_xfer_busy_cyc = 0;

  EngineModelConfig m_cfg{};
  OnChipConfig m_onchip{};
};

}  // namespace sim
}  // namespace moai

