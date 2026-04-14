#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "source/sim/engine_config.h"

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
    uint64_t last_finish_cycles = 0;
  };

  struct Summary {
    uint64_t makespan_cycles = 0;
    EngineStats dma{};
    EngineStats ntt{};
    EngineStats vec{};

    // On-chip transfer + capacity accounting (rough)
    EngineStats onchip_xfer{};
    EngineStats ntt_rf_xfer{};
    EngineStats vec_rf_xfer{};
    uint64_t gspad_bytes_used = 0;
    uint64_t gspad_spill_bytes = 0;
    uint64_t ntt_rf_spill_bytes = 0;
    uint64_t vec_rf_spill_bytes = 0;
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
    m_vec = {};
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
    s.vec = m_vec;
    s.onchip_xfer = m_onchip_xfer;
    s.ntt_rf_xfer = m_ntt_rf_xfer;
    s.vec_rf_xfer = m_vec_rf_xfer;
    s.gspad_bytes_used = m_gspad_used;
    s.gspad_spill_bytes = m_gspad_spill;
    s.ntt_rf_spill_bytes = m_ntt_rf_spill;
    s.vec_rf_spill_bytes = m_vec_rf_spill;
    return s;
  }

  void print_summary(std::ostream &os, const char *tag = "EngineModel") const {
    const Summary s = summary();
    os << "[MOAI_SIM_ENGINE] " << tag << " makespan_cycles=" << s.makespan_cycles << "\n";

    // Config sanity line (so logs are self-describing).
    const uint64_t gspad_eff_bw = std::max<uint64_t>(1, m_onchip.gspad_bw_gbps) * std::max<uint64_t>(1, m_onchip.gspad_banks);
    const uint64_t ntt_rf_eff_bw = std::max<uint64_t>(1, m_onchip.ntt_rf_bw_gbps) * std::max<uint64_t>(1, m_onchip.ntt_rf_banks);
    const uint64_t vec_rf_eff_bw = std::max<uint64_t>(1, m_onchip.vec_rf_bw_gbps) * std::max<uint64_t>(1, m_onchip.vec_rf_banks);
    os << "[MOAI_SIM_ENGINE] cfg"
       << " cycle_ns=" << m_cfg.cycle_ns
       << " dma_bw_gbps=" << m_cfg.dma_bw_gbps
       << " gspad=" << fmt_bytes_iec(m_onchip.gspad_bytes) << " banks=" << m_onchip.gspad_banks
       << " bw_gbps=" << m_onchip.gspad_bw_gbps << " eff_bw_gbps=" << gspad_eff_bw
       << " ntt_rf=" << fmt_bytes_iec(m_onchip.ntt_rf_bytes) << " banks=" << m_onchip.ntt_rf_banks
       << " bw_gbps=" << m_onchip.ntt_rf_bw_gbps << " eff_bw_gbps=" << ntt_rf_eff_bw
       << " vec_rf=" << fmt_bytes_iec(m_onchip.vec_rf_bytes) << " banks=" << m_onchip.vec_rf_banks
       << " bw_gbps=" << m_onchip.vec_rf_bw_gbps << " eff_bw_gbps=" << vec_rf_eff_bw
       << "\n";

    auto row = [&](const char *name, const EngineStats &e) {
      os << "[MOAI_SIM_ENGINE] "
         << std::left << std::setw(12) << name
         << std::right << std::setw(14) << e.busy_cycles
         << std::setw(14) << fmt_bytes_iec(e.bytes)
         << std::setw(16) << e.last_finish_cycles
         << "\n";
    };

    os << "[MOAI_SIM_ENGINE] "
       << std::left << std::setw(12) << "engine"
       << std::right << std::setw(14) << "busy_cycles"
       << std::setw(14) << "bytes"
       << std::setw(16) << "last_finish"
       << "\n";
    row("dma", s.dma);
    row("ntt", s.ntt);
    row("vec", s.vec);
    row("onchip_xfer", s.onchip_xfer);
    row("ntt_rf_xfer", s.ntt_rf_xfer);
    row("vec_rf_xfer", s.vec_rf_xfer);

    if (m_onchip.enabled())
    {
      os << "[MOAI_SIM_ENGINE] gspad_bytes_used=" << s.gspad_bytes_used
         << " gspad_spill_bytes=" << s.gspad_spill_bytes
         << " ntt_rf_spill_bytes=" << s.ntt_rf_spill_bytes
         << " vec_rf_spill_bytes=" << s.vec_rf_spill_bytes
         << "\n";
    }
  }

  // -------------------------
  // Primitive enqueues
  // -------------------------
  uint64_t enqueue_dma(uint64_t bytes, uint64_t dep_done_cycles = 0) {
    const uint64_t service = dma_service_cycles(bytes);
    return schedule(m_dma, service, bytes, dep_done_cycles);
  }

  // Backward-compatible alias (older call sites).
  uint64_t enqueue_mem(uint64_t bytes, uint64_t dep_done_cycles = 0) { return enqueue_dma(bytes, dep_done_cycles); }

  uint64_t enqueue_onchip_xfer(uint64_t bytes, uint64_t dep_done_cycles = 0) {
    const uint64_t service = onchip_service_cycles(bytes);
    return schedule(m_onchip_xfer, service, bytes, dep_done_cycles);
  }

  uint64_t enqueue_ntt_rf_xfer(uint64_t bytes, uint64_t dep_done_cycles = 0) {
    const uint64_t service =
        rf_service_cycles(bytes, m_onchip.ntt_rf_bw_gbps, m_onchip.ntt_rf_banks);
    return schedule(m_ntt_rf_xfer, service, bytes, dep_done_cycles);
  }

  uint64_t enqueue_vec_rf_xfer(uint64_t bytes, uint64_t dep_done_cycles = 0) {
    const uint64_t service =
        rf_service_cycles(bytes, m_onchip.vec_rf_bw_gbps, m_onchip.vec_rf_banks);
    return schedule(m_vec_rf_xfer, service, bytes, dep_done_cycles);
  }

  uint64_t enqueue_ntt_coeffs(uint64_t coeffs, uint64_t dep_done_cycles = 0) {
    const uint64_t service = m_cfg.ntt_pass_overhead_cyc + (m_cfg.ntt_cyc_per_coeff * coeffs);
    // bytes are accounted separately in MEM ops; still track "compute bytes" as 0.
    return schedule(m_ntt, service, 0, dep_done_cycles);
  }

  uint64_t enqueue_vec_coeffs(uint64_t coeffs, uint64_t cycles_per_coeff, uint64_t dep_done_cycles = 0) {
    const uint64_t service = m_cfg.vec_pass_overhead_cyc + (cycles_per_coeff * coeffs);
    return schedule(m_vec, service, 0, dep_done_cycles);
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
        t = enqueue_dma(bytes_ct_chunk + pt_dma, t);
        (void)ensure_gspad_capacity(bytes_ct_chunk + pt_dma);

        // GlobalSPAD -> NttRF stage (ct + maybe pt)
        const uint64_t pt_xfer = (m_pt_const_resident ? 0ULL : bytes_pt_chunk);
        t = stage_into_ntt_rf(bytes_ct_chunk + pt_xfer, t);

        // NTT fwd on chunk
        t = enqueue_ntt_coeffs(c, t);

        // ct to VecRF, VEC mul on chunk
        t = stage_into_vec_rf(bytes_ct_chunk, t);
        t = enqueue_vec_coeffs(c, m_cfg.vec_cyc_per_coeff, t);

        // ct back to NttRF, NTT inv on chunk
        t = stage_into_ntt_rf(bytes_ct_chunk, t);
        t = enqueue_ntt_coeffs(c, t);

        // RF -> GlobalSPAD -> DMA writeback (chunk)
        t = enqueue_onchip_xfer(bytes_ct_chunk, t);
        t = enqueue_dma(bytes_ct_chunk, t);

        done += c;
      }
    }
    else
    {
      t = enqueue_dma(bytes_ct + bytes_pt, t);
      t = enqueue_ntt_coeffs(coeffs, t);                          // fwd
      t = enqueue_vec_coeffs(coeffs, m_cfg.vec_cyc_per_coeff, t); // mul
      t = enqueue_ntt_coeffs(coeffs, t);                          // inv
      t = enqueue_dma(bytes_ct, t);                               // write back ct
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
    t = enqueue_dma(bytes_ct + bytes_ct, t);                    // read acc + tmp
    t = enqueue_vec_coeffs(coeffs, m_cfg.vec_cyc_per_coeff, t);  // add
    t = enqueue_dma(bytes_ct, t);                               // write acc
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
      t = enqueue_onchip_xfer(bytes_ct, t);
    t = enqueue_dma(bytes_ct, t);
    return t;
  }

  // ct×ct/bootstrapping skeleton ops (very rough). Key bytes are parametrized via env.
  uint64_t enqueue_rotate(uint64_t ct_size, uint64_t poly_degree, uint64_t limbs, uint64_t dep_done_cycles = 0) {
    const uint64_t coeffs = poly_degree * limbs;
    const uint64_t bytes_ct = ct_size * coeffs * sizeof(uint64_t);
    const uint64_t key_bytes = m_cfg.galois_key_bytes;
    uint64_t t = dep_done_cycles;
    t = enqueue_dma(bytes_ct + key_bytes, t);
    if (m_onchip.enabled())
      t = enqueue_onchip_xfer(bytes_ct + key_bytes, t);
    // model keyswitch-ish compute
    t = enqueue_ntt_coeffs(coeffs, t);
    t = enqueue_vec_coeffs(coeffs, m_cfg.vec_cyc_per_coeff, t);
    t = enqueue_dma(bytes_ct, t);
    return t;
  }

  uint64_t enqueue_relinearize(uint64_t ct_size, uint64_t poly_degree, uint64_t limbs, uint64_t dep_done_cycles = 0) {
    const uint64_t coeffs = poly_degree * limbs;
    const uint64_t bytes_ct = ct_size * coeffs * sizeof(uint64_t);
    const uint64_t key_bytes = m_cfg.relin_key_bytes;
    uint64_t t = dep_done_cycles;
    t = enqueue_dma(bytes_ct + key_bytes, t);
    if (m_onchip.enabled())
      t = enqueue_onchip_xfer(bytes_ct + key_bytes, t);
    t = enqueue_ntt_coeffs(coeffs, t);
    t = enqueue_vec_coeffs(coeffs, m_cfg.vec_cyc_per_coeff, t);
    t = enqueue_dma(bytes_ct, t);
    return t;
  }

  uint64_t enqueue_modswitch(uint64_t ct_size, uint64_t poly_degree, uint64_t limbs, uint64_t dep_done_cycles = 0) {
    const uint64_t coeffs = poly_degree * limbs;
    const uint64_t bytes_ct = ct_size * coeffs * sizeof(uint64_t);
    uint64_t t = dep_done_cycles;
    t = enqueue_vec_coeffs(coeffs, m_cfg.modswitch_cyc_per_coeff, t);
    if (m_onchip.enabled())
      t = enqueue_onchip_xfer(bytes_ct, t);
    t = enqueue_dma(bytes_ct, t);
    return t;
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

  uint64_t dma_service_cycles(uint64_t bytes) const {
    // bytes / (GB/s) = seconds; convert to ns; then to cycles using CYCLE_NS
    const uint64_t bw = m_cfg.dma_bw_gbps;
    if (bw == 0) return m_cfg.dma_base_cyc;
    // ns = bytes / (GB/s) * 1e9 / 1e9 = bytes / (GB/s)
    // since 1 GB/s = 1e9 B/s => 1 B takes 1e9/bw seconds => 1/bw ns.
    const uint64_t xfer_ns = (bytes + bw - 1) / bw;  // rough
    const uint64_t cyc_ns = std::max<uint64_t>(1, m_cfg.cycle_ns);
    const uint64_t xfer_cyc = (xfer_ns + cyc_ns - 1) / cyc_ns;
    return m_cfg.dma_base_cyc + xfer_cyc;
  }

  uint64_t onchip_service_cycles(uint64_t bytes) const {
    if (!m_onchip.enabled()) return 0;
    const uint64_t banks = std::max<uint64_t>(1, m_onchip.gspad_banks);
    const uint64_t bw = std::max<uint64_t>(1, m_onchip.gspad_bw_gbps * banks);
    const uint64_t xfer_ns = (bytes + bw - 1) / bw;
    const uint64_t cyc_ns = std::max<uint64_t>(1, m_cfg.cycle_ns);
    const uint64_t xfer_cyc = (xfer_ns + cyc_ns - 1) / cyc_ns;
    return m_onchip.gspad_base_cyc + xfer_cyc;
  }

  uint64_t rf_service_cycles(uint64_t bytes, uint64_t rf_bw_gbps, uint64_t rf_banks) const {
    if (!m_onchip.enabled()) return 0;
    const uint64_t banks = std::max<uint64_t>(1, rf_banks);
    const uint64_t bw = std::max<uint64_t>(1, rf_bw_gbps * banks);
    const uint64_t xfer_ns = (bytes + bw - 1) / bw;
    const uint64_t cyc_ns = std::max<uint64_t>(1, m_cfg.cycle_ns);
    const uint64_t xfer_cyc = (xfer_ns + cyc_ns - 1) / cyc_ns;
    return 1 + xfer_cyc; // small base for local staging
  }

  uint64_t stage_into_ntt_rf(uint64_t bytes, uint64_t dep_done_cycles) {
    if (m_onchip.ntt_rf_bytes == 0) {
      // If not specified, fall back to generic onchip_xfer.
      return enqueue_onchip_xfer(bytes, dep_done_cycles);
    }
    if (bytes <= m_onchip.ntt_rf_bytes) {
      return enqueue_ntt_rf_xfer(bytes, dep_done_cycles);
    }
    // Stream in chunks; record spill as the bytes beyond capacity.
    m_ntt_rf_spill += (bytes - m_onchip.ntt_rf_bytes);
    uint64_t t = dep_done_cycles;
    uint64_t remaining = bytes;
    while (remaining) {
      const uint64_t chunk = std::min<uint64_t>(remaining, m_onchip.ntt_rf_bytes);
      t = enqueue_ntt_rf_xfer(chunk, t);
      remaining -= chunk;
    }
    return t;
  }

  uint64_t stage_into_vec_rf(uint64_t bytes, uint64_t dep_done_cycles) {
    if (m_onchip.vec_rf_bytes == 0) {
      return enqueue_onchip_xfer(bytes, dep_done_cycles);
    }
    if (bytes <= m_onchip.vec_rf_bytes) {
      return enqueue_vec_rf_xfer(bytes, dep_done_cycles);
    }
    m_vec_rf_spill += (bytes - m_onchip.vec_rf_bytes);
    uint64_t t = dep_done_cycles;
    uint64_t remaining = bytes;
    while (remaining) {
      const uint64_t chunk = std::min<uint64_t>(remaining, m_onchip.vec_rf_bytes);
      t = enqueue_vec_rf_xfer(chunk, t);
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

  uint64_t schedule(EngineStats &e, uint64_t service_cycles, uint64_t bytes, uint64_t dep_done_cycles) {
    const uint64_t start = std::max(e.last_finish_cycles, dep_done_cycles);
    const uint64_t finish = start + service_cycles;
    e.last_finish_cycles = finish;
    e.busy_cycles += service_cycles;
    e.bytes += bytes;
    m_makespan = std::max(m_makespan, finish);
    return finish;
  }

  EngineStats m_dma{};
  EngineStats m_ntt{};
  EngineStats m_vec{};
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

  EngineModelConfig m_cfg{};
  OnChipConfig m_onchip{};
};

}  // namespace sim
}  // namespace moai

