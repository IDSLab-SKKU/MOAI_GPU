## OTF eval-key `a` PRNG engine (simulator)

### 목적
- eval-key를 (b,a)로 보고, **랜덤 다항식 `a`** 는 외부 메모리에서 읽지 않고 **SHAKE128 + reject sampling** 으로 생성한다고 가정한다.
- `a`는 각 RNS limb modulus \(q_j\) 에 대해 **균일 분포**여야 하며, NTT는 bijection이므로 **NTT-domain에서 직접 샘플링한 것으로 간주**해 downstream `evk Mult`가 추가 NTT 없이 소비한다고 모델링한다.

### 구현 위치
- SHAKE128 + reject sampling + descriptor/도메인 분리 + timing 모델: `src/include/source/sim/prng_evalkey_a.h`
- 마이크로벤치/유닛테스트: `src/include/test/sim/test_sim_prng_evalkey_a.cuh`
- 벤치 엔트리: `src/test.cu` (`MOAI_BENCH_MODE=sim_prng_evalkey_a`)

### 주요 env knobs
- (기존) `MOAI_SIM_OTF_KEYGEN=1`: on-the-fly keygen 활성화
- 상세 PRNG timing:
  - `MOAI_SIM_OTF_EVK_A_NUM_PRNG_LANES` (default 4)
  - `MOAI_SIM_OTF_EVK_A_NUM_SAMPLER_LANES` (default 4)
  - `MOAI_SIM_OTF_EVK_A_SHAKE_STARTUP_CYCLES` (default 50)
  - `MOAI_SIM_OTF_EVK_A_SHAKE_BLOCK_CYCLES` (default 4)
  - `MOAI_SIM_OTF_EVK_A_BIT_FIFO_CAPACITY_BLOCKS` (default 8)
  - `MOAI_SIM_OTF_EVK_A_COEFF_FIFO_CAPACITY` (default 256)
  - `MOAI_SIM_OTF_EVK_A_CHUNK_SIZE` (default 32)
  - `MOAI_SIM_OTF_EVK_A_SEED_METADATA_BYTES` (default 64)

### 측정 가능한 것(요약)
- `stall_cycles_due_to_prng_unavailability`: consumer가 `a` chunk를 필요로 했는데 준비되지 않아 노출된 stall.
- `first_request_cycle`, `first_needed_cycle`, `first_ready_cycle`, `last_ready_cycle`: overlap(숨김/노출) 판단용 타임스탬프.

### 실행
```bash
cmake --build MOAI_GPU/build -j"$(nproc)"

# 유닛 테스트 + 마이크로벤치(타이밍 모델 출력)
MOAI_BENCH_MODE=sim_prng_evalkey_a MOAI_TEST_OUTPUT_DISABLE=1 ./MOAI_GPU/build/test
```

### 한계 / 러프한 부분
- timing은 cycle-정확 FIFO 시뮬이 아니라, **throughput + bounded buffering를 근사한 모델**이다.
- reject sampling accept 확률은 `accept_prob_hint` 기반(기본 1.0)이며, 큰 q에 대한 정확한 \(T/2^{64}\)는 향후 limb modulus를 실제로 주입하면 더 정확히 모델링 가능.

