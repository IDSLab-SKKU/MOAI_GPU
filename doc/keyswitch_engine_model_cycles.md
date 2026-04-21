# Phantom CKKS 키스위치: EngineModel 사이클·연산량 수식

이 문서는 `MOAI_GPU` 시뮬레이터에서 **Phantom** `eval_key_switch.cu` / `rns_bconv.cu` (CKKS keyswitch)에 맞춘 `EngineModel::enqueue_keyswitch_phantom_ckks`가 **NTT/INTT, BConv, vec_mul/add**에 대해 어떤 **수식**으로 `busy_cycles`·`logical_ops`·MMUL/MADD 카운트를 내는지 정리한다.

구현 기준 파일:

- `src/include/source/sim/engine_model.h` — 스케줄·사이클
- `src/include/source/sim/engine_config.h` — env로 조절되는 상수
- `src/include/source/sim/keyswitch_op_profile.h` — 동일 그래프의 analytic mirror (CSV)

---

## 1. 기호

| 기호 | 의미 |
|------|------|
| \(N\) | `poly_degree` (다항식 차수, coeff 개수) |
| \(\|Ql\|\) | `limbs` — 데이터 레벨에서 ciphertext의 Q base 소수 개수 |
| \(\|P\|\) | `num_P` = `kswitch_size_p` — special modulus 개수 (Phantom에서 보통 \(\alpha\)와 동일) |
| \(\alpha\) | hybrid digit 크기 = `MOAI_SIM_ALPHA` → `kswitch_digit_alpha` |
| \(\|QlP\|\) | \(\|Ql\| + \|P\|\) |
| \(\beta\) | digit 개수. Phantom: \(\beta = \lceil \|Ql\| / \alpha \rceil\) (`keyswitch_beta_choose`) |

한 keyswitch 호출에서 **coeff-element**란 \(N \times (\text{해당 패스의 limb 수})\) 개의 uint64 계수를 말한다.

---

## 2. 스케줄러 공통

엔진은 의존성 그래프 위에서 `schedule(engine, service_cycles, ...)`로 완료 시각을 갱신한다.

- **`eng_*_busy_cyc`**: 해당 엔진에 누적된 `service_cycles` 합 (파이프라인 겹침과 무관하게 **엔진별 합**).
- **`eng_makespan_cyc`**: 크리티컬 패스 상의 wall-clock 근사 (여러 엔진이 겹치면 합이 makespan보다 클 수 있음).

**VEC 파이프 분리:** `vec_bconv` = 키스위치 **modup/moddown base conversion**만. `vec_arith` = 그 외 모든 벡터 연산(**vec_mul, vec_add, enqueue_vec_mac** 등). 융합 MAC은 `vec_arith`의 `busy_cycles`에 포함되고, 세부량은 `vec_arith.mac_ops`로 본다.

---

## 3. NTT / INTT (`enqueue_ntt_coeffs`)

**입력:** `coeffs` = 이번 패스에서 처리하는 coeff-element 개수, `poly_degree` = \(N\), `ntt_inverse` = fwd vs inv 구분만 통계용.

**서비스 사이클:**

\[
\text{ntt\_service} =
\text{ntt\_pass\_overhead} +
\text{fill} +
\underbrace{\left\lceil \frac{\text{coeffs}}{\text{ntt\_lanes}} \right\rceil}_{\text{waves}} \cdot \text{ntt\_steady\_cyc\_per\_coeff}
\]

구현(`ntt_service_cycles`)은 `waves * ntt_steady_cyc_per_coeff`로 곱한다. 필드 이름은 `*_per_coeff`이지만, 실제로는 **wavefront 스텝당** steady 비용으로 쓰는 것이 맞고(주석 참고: `engine_config.h`), 레인당 한 번에 `lanes`개 coeff를 밀어 넣는 거친 모델이다.

- **fill** (`ntt_pipe_fill_cycles`):
  - `MOAI_SIM_NTT_PIPE_DEPTH_CYC > 0`이면 그 값.
  - 아니면 \(\lfloor \log_2 N \rfloor\) (예: \(N=65536 \Rightarrow 16\)).

**통계:**

- `logical_ops` (NTT/INTT 합계 및 fwd/inv 분리) += `coeffs`
- `busy_cycles` += `ntt_service`

**환경 변수 (요약):** `MOAI_SIM_NTT_LANES`, `MOAI_SIM_NTT_STEADY_CYC_PER_COEFF` (미설정 시 레거시 `MOAI_SIM_NTT_CYC_PER_COEFF`로 대체), `MOAI_SIM_NTT_PIPE_DEPTH_CYC`, `MOAI_SIM_NTT_PASS_OVERHEAD_CYC`.

---

## 4. Keyswitch 그래프 (`enqueue_keyswitch_phantom_ckks`)

`limbs` = \(\|Ql\|\), `num_P` = \(\|P\|\), `size_QlP` = `limbs + num_P`, `beta` = digit 수, `alpha` = digit 크기.

실제 enqueue 순서(요약):

1. **INTT on \(Ql\)**: `coeffs_ql = N * |Ql|`
2. **modup 루프** `bi = 0 .. beta-1`:
   - **BConv** (modup): 아래 §5
   - **NTT fwd** on `QlP`에서 해당 digit의 part-\(Ql\) 구간 **제외**한 coeff 수:
     - `coeffs_excl_range = N * (size_QlP - part_limbs)`
3. **inner product proxy**: `vec_mul` × `beta`번, 각 `coeffs_qlp = N * |QlP|`
4. **moddown × 2** (각 성분):
   - **INTT** on \(P\) only: `coeffs_p = N * |P|`
   - **BConv** (moddown): 아래 §5
   - **NTT fwd** on \(Ql\) (fuse moddown 모델): `coeffs_ql`
   - **vec_mul** on \(Ql\)
5. **vec_add** × 2 on \(Ql\)

Phantom 구현과의 대응은 `md/hybrid_ks_profile_worklog.md` 및 Phantom 소스 주석을 참고한다.

---

## 5. BConv (base conversion)

### 5.1 레거시 모드 (상수 cycles/coeff)

`MOAI_SIM_KSWITCH_BCONV_USE_MONTGOMERY_OPS=0` (기본):

- modup: `enqueue_vec_coeffs(coeffs_qlp, kswitch_modup_bconv_cyc_per_coeff, ...)`
- moddown: `enqueue_vec_coeffs(coeffs_ql, kswitch_moddown_bconv_cyc_per_coeff, ...)`

\[
\text{vec\_service} =
\text{vec\_pass\_overhead} +
\left\lceil \frac{\text{coeffs}}{\text{vec\_lanes}} \right\rceil \cdot \text{cycles\_per\_coeff}
\]

이 모드에서는 **matmul 차원이 사이클에 직접 들어가지 않고**, “한 스케줄당 coeff 스트림 길이”만 반영된다. 보고용으로는 여전히 MMUL/MADD op 수를 누적할 수 있다 (`keyswitch_op_profile` 및 엔진의 `mmul_ops`/`madd_ops`).

**환경:** `MOAI_SIM_KSWITCH_MODUP_BCONV_CYC_PER_COEFF`, `MOAI_SIM_KSWITCH_MODDOWN_BCONV_CYC_PER_COEFF`.

### 5.2 Montgomery / matmul 모드 (권장)

`MOAI_SIM_KSWITCH_BCONV_USE_MONTGOMERY_OPS=1`:

Phantom BEHZ BConv를 **out-limb × in-limb** matmul의 MMUL/MADD로 보고, 사이클은 다음으로 잡는다.

\[
\text{bconv\_service} =
\text{vec\_pass\_overhead} +
\left\lceil \frac{\text{mmul\_ops}}{\text{vec\_lanes}} \right\rceil \cdot \text{vec\_mmul\_cyc\_per\_op} +
\left\lceil \frac{\text{madd\_ops}}{\text{vec\_lanes}} \right\rceil \cdot \text{vec\_madd\_cyc\_per\_op}
\]

- **final reduction**(Barrett 등)은 별도 항으로 두지 않고 **MMUL/MADD에 흡수**했다고 가정한다.

**환경:** `MOAI_SIM_VEC_MMUL_CYC_PER_OP`, `MOAI_SIM_VEC_MADD_CYC_PER_OP`, `MOAI_SIM_VEC_LANES`, `MOAI_SIM_VEC_PASS_OVERHEAD_CYC`.

### 5.3 FMA / MAC 모드 (벡터 엔진에 융합 곱셈–누적이 있을 때)

`MOAI_SIM_KSWITCH_BCONV_USE_MONTGOMERY_OPS=1`인 것과 **함께** `MOAI_SIM_KSWITCH_BCONV_USE_FMA_OPS=1`이면, BConv 사이클을 **MMUL 파형 + MADD 파형을 따로 더하지 않고**, 아래처럼 잡는다 (`vec_bconv_montgomery_service_cycles`).

- **MAC(융합) 연산 수:** `mac_ops = madd_ops` — 내적·누적 쪽을 한 종류의 FMA로 처리.
- **곱만 있는 추가 연산:** `extra_mul_ops = max(0, mmul_ops − madd_ops)` — BEHZ **moddown**에서 `mmul − madd = N·|P|`인 **phase 1**은 FMA가 아니라 **스케일 곱**으로 남는 부분이다 (`rns_bconv.cu`의 phase1 `bconv_mult_*` vs phase2 `bconv_matmul_*`).

\[
\text{bconv\_service}_{\text{FMA}} =
\text{vec\_pass\_overhead} +
\left\lceil \frac{\text{mac\_ops}}{\text{vec\_lanes}} \right\rceil \cdot \text{vec\_mac\_cyc\_per\_op} +
\left\lceil \frac{\text{extra\_mul\_ops}}{\text{vec\_lanes}} \right\rceil \cdot \text{vec\_mmul\_cyc\_per\_op}
\]

- **modup**은 `mmul_ops = madd_ops`이므로 `extra_mul_ops = 0` — 전부 MAC wave로만 스케줄된다 (기존 분리 모델 대비 대략 **한 파이프 분**만큼 덜 보수적).
- CSV·통계에 쌓이는 `mmul_ops` / `madd_ops` **숫자는 그대로**이고, 바뀌는 것은 **스케줄에 쓰는 사이클**뿐이다.

**환경:** `MOAI_SIM_KSWITCH_BCONV_USE_FMA_OPS`, `MOAI_SIM_VEC_MAC_CYC_PER_OP` (미설정 시 `MOAI_SIM_VEC_MMUL_CYC_PER_OP`와 동일하게 로드).

#### modup (digit `bi`)

- `part_limbs` = 해당 digit의 Ql 조각 크기 (마지막 digit은 잔여 포함).
- `ibase_size` = `part_limbs`
- `obase_size` = `size_QlP - part_limbs` (complement)

\[
\text{mmul} = \text{madd} = N \cdot \text{obase\_size} \cdot \text{ibase\_size}
\]

#### moddown (한 번, keyswitch에서 2회 호출)

- `ibase_size` = `num_P` = \(\|P\|\)
- `obase_size` = `limbs` = \(\|Ql\|\)

BEHZ 2단계를 반영한 합:

\[
\text{mmul} = N \cdot (\|P\| + \|Ql\| \cdot \|P\|)
\]
\[
\text{madd} = N \cdot (\|Ql\| \cdot \|P\|)
\]

(phase1: in-base 스케일 곱 \(\|P\|\)개 + phase2: matmul \(\|Ql\|\times\|P\|\)에서 곱·누적.)

---

## 6. vec_mul / vec_add (keyswitch 내부)

### vec_mul

\[
\text{service} =
\text{vec\_pass\_overhead} +
\left\lceil \frac{\text{coeffs}}{\text{vec\_lanes}} \right\rceil \cdot \text{vec\_mul\_steady\_cyc\_per\_coeff}
\]

통계: `vec_arith.mmul_ops += coeffs` (elementwise modular mul 프록시).

**환경:** `MOAI_SIM_VEC_MUL_STEADY_CYC_PER_COEFF` (없으면 `MOAI_SIM_VEC_MUL_CYC_PER_COEFF`, 그다음 `MOAI_SIM_VEC_CYC_PER_COEFF`), `MOAI_SIM_VEC_LANES`, `MOAI_SIM_VEC_PASS_OVERHEAD_CYC`.

### vec_add

`enqueue_vec_coeffs`와 동일한 wave 모델에 `vec_add_cyc_per_coeff` 사용 후, `madd_ops += coeffs`.

**환경:** `MOAI_SIM_VEC_ADD_CYC_PER_COEFF`.

### vec_mul / vec_add 외: `enqueue_vec_mac` (융합 MAC)

같은 VEC 파이프(`m_vec`)에서 **계수 스트림에 대해 곱–누적 한 번에** 처리하는 경로다. 사이클은 `vec_mul`과 동일한 wave 모델이며 steady는 `vec_mac_steady_cyc_per_coeff`를 쓴다.

- 통계는 **`Summary::vec_arith`** 에 합산된다 (`busy_cycles`, `logical_ops`). 융합 MAC만 **`vec_arith.mac_ops`** += `coeffs`로 따로 본다 (BConv 전용 `vec_bconv`와 구분).
- 키스위치 기본 그래프(`enqueue_keyswitch_phantom_ckks`)는 **vec_mul / vec_add**만 쓰는 경우가 많고, MAC을 쓰면 같은 **arith** 버킷에 쌓인다.

**환경:** `MOAI_SIM_VEC_MAC_STEADY_CYC_PER_COEFF` (미설정 시 `MOAI_SIM_VEC_MAC_CYC_PER_COEFF`, 그다음 `vec_mul_steady`와 동일하게 로드).

---

## 7. Analytic mirror (`keyswitch_op_profile.h`)

`compute_keyswitch_phantom_profile`은 위 그래프와 동일한 **가중 coeff-element 합**과 **BConv MMUL/MADD 합**을 CSV용으로 계산한다.

예:

- **NTT fwd 가중 합:** \(\beta \cdot N|QlP| - N|Ql| + 2 \cdot N|Ql|\) (modup에서 제외 누적 + moddown fuse 2회)
- **INTT 가중 합:** \(N|Ql| + 2 \cdot N|P|\)
- **BConv modup/moddown** MMUL/MADD는 §5.2와 동일 식을 digit 루프/2회 moddown에 대해 합산.

**주의:** analytic에서 digit 크기를 `alpha = max(1, num_P)`로 두고 `part_limbs`를 나눈다. 엔진의 `enqueue_keyswitch_phantom_ckks`는 `kswitch_digit_alpha` (`MOAI_SIM_ALPHA` 등)를 쓴다. **한 스윕에서 `num_P`와 digit `alpha`가 같게 맞춰야** modup BConv op 수가 analytic과 `EngineModel`이 일치한다.

`sim_hybrid_ks_profile` 벤치가 이 헤더를 통해 `hybrid_ks_profile.csv`를 채운다.

---

## 8. 스윕 모드 두 가지

| 목적 | 대표 설정 |
|------|-----------|
| **\(T=\|QP\|\) 고정** (일반 체인 스윕) | `MOAI_SIM_NUM_LIMBS` = \|QP\|, `size_Ql` ≈ `T - alpha` (기본) |
| **\|Ql\| 고정** (논문처럼 Q만 고정) | `MOAI_SIM_KSWITCH_SIZE_Ql` = 상수, `MOAI_SIM_NUM_LIMBS`는 \|Ql\|+\|P\| 표시용 등으로 맞춤 |

BConv 사이클을 matmul과 일치시키려면 **`MOAI_SIM_KSWITCH_BCONV_USE_MONTGOMERY_OPS=1`** 권장.

---

## 9. 요약 표

| 구성요소 | logical_ops (coeff-elem) | busy_cycles 모델 |
|----------|---------------------------|------------------|
| NTT/INTT | `coeffs` | §3 파이프라인 + wave |
| BConv (레거시) | `coeffs` (enqueue 길이) | coeff당 상수 × wave |
| BConv (Montgomery) | 동일 | §5.2 MMUL/MADD wave |
| vec_mul | `coeffs` | mul steady × wave |
| vec_add | `coeffs` | add cyc × wave |

---

## 10. 시뮬레이터 조건 설정·CSV 생성·그래프

아래는 `sim_hybrid_ks_profile` 벤치(`keyswitch_op_profile.h`의 `moai_sim_hybrid_ks_profile_run`)로 **α 스윕 CSV**를 만들고, `plot_hybrid_ks_profile.py`로 **PNG**를 뽑는 절차다. 저장소 루트는 `MOAI_GPU/`로 둔다.

### 10.1 빌드

`./build/test`가 있어야 한다 (CMake 타깃은 프로젝트 설정에 따름).

```bash
cd /path/to/MOAI_GPU
cmake --build build
```

### 10.2 CSV 한 번 생성 (필수 env)

벤치 모드로 `test` 바이너리를 실행한다.

```bash
export MOAI_TEST_OUTPUT_DISABLE=1
export MOAI_BENCH_MODE=sim_hybrid_ks_profile
./build/test
```

- 기본 CSV 경로: `output/sim/hybrid_ks_profile.csv` (`MOAI_SIM_HYBRID_KS_PROFILE_CSV`로 변경 가능).
- 부모 디렉터리가 없으면 벤치가 생성을 시도한다.

### 10.3 α 스윕·체인 크기·엔진 계측 (대표 env)

| 목적 | 환경 변수 (예) |
|------|----------------|
| α 목록 | `MOAI_SIM_HYBRID_KS_ALPHA_RANGE=1-35` 또는 `MOAI_SIM_HYBRID_KS_ALPHA_LIST=1,2,4,8` (LIST가 있으면 RANGE보다 우선) |
| “정확히 나눠지는 digit”만 쓰기 | `MOAI_SIM_HYBRID_KS_EXACT_PARTITION=1` (기본): `size_Ql ≥ α` 이고 `size_Ql mod α ≠ 0` 인 행 스킵 |
| 모든 α 포함 | `MOAI_SIM_HYBRID_KS_EXACT_PARTITION=0` |
| 전체 체인 길이 `T_chain` (CSV `t_qp` ≈ 상위 **QP** 길이; `alpha < T_chain` 필요) | `MOAI_SIM_NUM_LIMBS=36` 등 |
| 다항식 차수 `N` | `MOAI_SIM_POLY_DEGREE` (미설정 시 싱글레이어 기본값) |
| **Q 크기 고정** (`size_Ql`; 논문식으로 Q만 고정하고 α만 변화) | `MOAI_SIM_KSWITCH_SIZE_Ql=35` — 행마다 `size_Ql`이 이 값으로 고정 |
| **P 크기 고정** (기본은 행마다 `num_P = α`) | `MOAI_SIM_KSWITCH_SIZE_P`를 명시하면 `num_P`가 고정 (analytic의 digit `part`와 엔진 `MOAI_SIM_ALPHA` 불일치 주의) |
| β 강제 | `MOAI_SIM_KSWITCH_BETA` (>0이면 모든 행에 동일 β) |
| β 규칙 | `MOAI_SIM_KSWITCH_BETA_MODE=phantom` (기본) 또는 `legacy` |
| **엔진 스케줄·`eng_*` busy 사이클**까지 CSV에 쓰기 | `MOAI_SIM_BACKEND=1` + `MOAI_SIM_HYBRID_KS_MEASURE_ENGINE=1` |
| **BConv 사이클을 MMUL/MADD op 기반으로** (`engine_model.h` §5.2) | `MOAI_SIM_KSWITCH_BCONV_USE_MONTGOMERY_OPS=1` (+ `MOAI_SIM_VEC_MMUL_CYC_PER_OP`, `MOAI_SIM_VEC_MADD_CYC_PER_OP`, `MOAI_SIM_VEC_LANES` 등) |
| **BConv를 FMA/MAC + (필요 시) 추가 MMUL**로 덜 보수적으로 | `MOAI_SIM_KSWITCH_BCONV_USE_FMA_OPS=1` (Montgomery와 함께), 선택 `MOAI_SIM_VEC_MAC_CYC_PER_OP` (§5.3) |

행별로 벤치가 `MOAI_SIM_ALPHA`를 덮어쓴 뒤 `EngineModel::profile_keyswitch_phantom_ckks(N, size_Ql, …)`를 호출하므로, **스윕 시 `num_P`는 기본적으로 그 행의 α**이고, `MOAI_SIM_KSWITCH_SIZE_Ql`이 0이면 `size_Ql = T_chain - α`가 된다.

### 10.4 권장 원샷: 스크립트로 CSV + 전체 플롯

`run_hybrid_ks_sweep_plots.sh`는 CSV 생성 후 `--all-plots`로 여러 PNG를 한 번에 쓴다.

```bash
cd /path/to/MOAI_GPU
src/scripts/run_hybrid_ks_sweep_plots.sh
```

스크립트 기본:

- `MOAI_SIM_HYBRID_KS_ALPHA_RANGE=1-35`
- `MOAI_SIM_HYBRID_KS_EXACT_PARTITION=1`
- `MOAI_SIM_BACKEND=1`, `MOAI_SIM_HYBRID_KS_MEASURE_ENGINE=1`

env로 덮어쓰기 예:

```bash
MOAI_SIM_HYBRID_KS_EXACT_PARTITION=0 \
MOAI_SIM_KSWITCH_BCONV_USE_MONTGOMERY_OPS=1 \
src/scripts/run_hybrid_ks_sweep_plots.sh
```

### 10.5 CSV만 따로 뽑은 뒤 플롯만 돌리기

```bash
MOAI_TEST_OUTPUT_DISABLE=1 MOAI_BENCH_MODE=sim_hybrid_ks_profile \
  MOAI_SIM_BACKEND=1 MOAI_SIM_HYBRID_KS_MEASURE_ENGINE=1 \
  MOAI_SIM_HYBRID_KS_ALPHA_RANGE=1-35 \
  ./build/test

python3 src/scripts/plot_hybrid_ks_profile.py --all-plots --csv output/sim/hybrid_ks_profile.csv
```

`--all-plots`는 내부적으로 스택 바, 엔진 비교, 메모리, **사이클 스택/라인**, 도넛, modulus 크기 플롯·텍스트, **Montgomery cycles** 등을 켠다 (`plot_hybrid_ks_profile.py` 상단 docstring 참고).

개별 옵션만 쓰려면 예:

- `--compare-engine` / `--cycles-plots` / `--pie` / `--memory` / `--mod-sizes` / `--mod-sizes-txt` / `--montgomery-cycles`
- 출력 경로: `--out`, `--out-compare`, `--out-cycles-stacked`, `--out-cycles-lines`, `--out-pie`, `--out-montgomery-cycles` 등

### 10.6 대표 출력 파일

| 산출물 | 기본 경로 (변경 가능) |
|--------|----------------------|
| CSV | `output/sim/hybrid_ks_profile.csv` |
| 구조적 스택(커널/가중) | `output/sim/hybrid_ks_profile_stacked.png` |
| 엔진 vs analytic 비교 | `output/sim/hybrid_ks_profile_engine_compare.png` |
| busy 사이클 스택 / 라인 | `hybrid_ks_profile_stacked_cycles.png`, `hybrid_ks_profile_engine_cycles.png` |
| MMUL/MADD 기반 VEC 사이클 추정 | `hybrid_ks_profile_montgomery_cycles.png` (`--mmul-cyc`, `--madd-cyc`로 유효 사이클/스루풋 반영) |
| 메모리 | `hybrid_ks_profile_memory.png` |
| 도넛 % | `hybrid_ks_profile_pie.png` |
| T·BTS 고정 modulus 표/그래프 | `hybrid_ks_profile_mod_sizes.txt`, `hybrid_ks_profile_mod_sizes.png` |

`eng_*` 사이클·도넛의 시뮬레이터 비율은 **CSV에 `eng_makespan_cyc` 등이 -1이 아닐 때** 의미가 있다. 없으면 스크립트가 재실행을 안내한다.

### 10.7 `size_Ql` 고정 스윕 예 (별도 CSV 권장)

기본 파일명을 덮어쓰지 않으려면 `MOAI_SIM_HYBRID_KS_PROFILE_CSV`로 분리한다.

```bash
export MOAI_TEST_OUTPUT_DISABLE=1
export MOAI_BENCH_MODE=sim_hybrid_ks_profile
export MOAI_SIM_BACKEND=1
export MOAI_SIM_HYBRID_KS_MEASURE_ENGINE=1
export MOAI_SIM_KSWITCH_SIZE_Ql=35
export MOAI_SIM_NUM_LIMBS=70
export MOAI_SIM_HYBRID_KS_EXACT_PARTITION=0
export MOAI_SIM_HYBRID_KS_ALPHA_RANGE=1-34
export MOAI_SIM_KSWITCH_BCONV_USE_MONTGOMERY_OPS=1
export MOAI_SIM_HYBRID_KS_PROFILE_CSV=output/sim/hybrid_ks_profile_fixedQ35.csv

./build/test

python3 src/scripts/plot_hybrid_ks_profile.py --all-plots --csv output/sim/hybrid_ks_profile_fixedQ35.csv \
  --out output/sim/hybrid_ks_profile_fixedQ35_stacked.png \
  --out-compare output/sim/hybrid_ks_profile_fixedQ35_engine_compare.png \
  --out-cycles-stacked output/sim/hybrid_ks_profile_fixedQ35_stacked_cycles.png \
  --out-cycles-lines output/sim/hybrid_ks_profile_fixedQ35_engine_cycles.png \
  --out-montgomery-cycles output/sim/hybrid_ks_profile_fixedQ35_montgomery_cycles.png
```

(`--all-plots`는 여러 `--out-*` 기본값을 쓰므로, 위처럼 **같은 접두어로 맞춘 파일명**을 주면 결과가 섞이지 않는다.)

상세 팁·Phantom 정합 설명은 `md/hybrid_ks_profile_worklog.md`를 병행한다.

---

## 11. 참고

- Phantom 코드: `thirdparty/phantom-fhe/src/eval_key_switch.cu`, `rns_bconv.cu`, `rns.cu`
- 작업 로그: `md/hybrid_ks_profile_worklog.md`
- 플롯 CLI: `src/scripts/plot_hybrid_ks_profile.py`, 일괄 실행: `src/scripts/run_hybrid_ks_sweep_plots.sh`
