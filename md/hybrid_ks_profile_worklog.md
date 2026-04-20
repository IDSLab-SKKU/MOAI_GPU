# Hybrid KS profile worklog (MOAI\_GPU)

이 문서는 `sim_hybrid_ks_profile` 스윕/플롯/엔진계측 및 Phantom(hybrid keyswitch) 코드 정합을 위해 수행한 변경 사항을 요약한다.

## 목표

- **α sweep 정책**
  - 기본: `|Ql| ≥ α` 이고 `|Ql| mod α ≠ 0` 이면 해당 α를 **스윕에서 제외** (digit partition이 “정확히” 떨어지지 않는 경우).
  - 옵션: 모든 α 포함은 `MOAI_SIM_HYBRID_KS_EXACT_PARTITION=0`.
- **BConv 분리 집계/플롯**
  - BConv는 물리적으로 VEC 파이프에서 돌지만, 통계/플롯에서는 **`vec_bconv` vs `vec_arith`(mul/add)**로 분리해 보고 싶음.
- **Phantom 구현과 NTT/INTT 수식 정합**
  - Phantom `modup`/`moddown_from_NTT` 최적화(exclude-range, P-only INTT, Ql-only fused forward)를 시뮬레이터(`EngineModel`)와 analytic 수식(`keyswitch_op_profile`)에 반영.

---

## 변경 사항 요약

### 1) `engine_model.h`: VEC 통계 BConv vs arith 분리

- VEC 스케줄은 기존처럼 `m_vec`에 쌓이되, keyswitch의 base-conversion(BConv) 경로만 **별도 통계**로 분리.
- `Summary`에 `vec_bconv`, `vec_arith` 추가. 출력/리셋/요약에 반영.

관련 파일:
- `src/include/source/sim/engine_model.h`

### 2) `keyswitch_op_profile.h`: CSV에 `eng_vec_bconv_*`/`eng_vec_arith_*` 추가 + 출력 정합

- CSV 헤더/행에 아래 컬럼 추가:
  - `eng_vec_bconv_enq`, `eng_vec_arith_enq`
  - `eng_vec_bconv_coeff_ops`, `eng_vec_arith_coeff_ops`
  - `eng_vec_bconv_busy_cyc`, `eng_vec_arith_busy_cyc`
- `eng_*` 계측이 켜졌을 때 `EngineModel::Summary`의 split stats를 읽어서 CSV에 기록.
- 누락되기 쉬운 `int64_t ... = -1` 초기화/CSV 출력 순서까지 맞춤.

관련 파일:
- `src/include/source/sim/keyswitch_op_profile.h`

### 3) `plot_hybrid_ks_profile.py`: 스택/엔진비교/사이클/도넛(%)

- **스택(bar)**: 5단 분해
  - `NTT fwd` / `INTT` / `BConv` / `vec_mul` / `vec_add`
- **엔진 비교(2×2)** (가능할 때)
  - `NTT fwd`, `INTT`, `VEC BConv`(engine `vec_bconv`), `VEC arith`(engine `vec_arith`)
- **사이클 플롯**
  - `eng_vec_bconv_busy_cyc` / `eng_vec_arith_busy_cyc`가 있으면 사이클도 BConv/arith로 split 표시
- **도넛 차트(%)**
  - `--pie` 또는 `--all-plots`에서 생성
  - 구조적(workload sum) % + (가능하면) simulator busy\_cycles %를 같은 그림에 출력
  - 기본 출력: `output/sim/hybrid_ks_profile_pie.png`

관련 파일:
- `src/scripts/plot_hybrid_ks_profile.py`

### 4) `run_hybrid_ks_sweep_plots.sh`: EXACT_PARTITION 기본값 복구

- 사용자가 원했던 정책(“나눠지지 않으면 제외”)에 맞게 스크립트 기본을 **`MOAI_SIM_HYBRID_KS_EXACT_PARTITION=1`**로 설정.
- 모든 α 포함을 원하면 실행 전 env로 `0`을 주면 됨.

관련 파일:
- `src/scripts/run_hybrid_ks_sweep_plots.sh`

---

## Phantom 코드 기준: 도메인/NTT·INTT 해석 (CKKS)

### 입력 도메인 가정

Phantom `DRNSTool::modup()`는 주석/코드로 **입력 `cks`가 NTT 도메인**임을 가정한다.

- `cks` in NTT domain → base conversion을 위해 잠시 normal로 내려서(bwd) 계산 → 다시 fwd로 올려 inner-product/moddown에 씀.

참고 코드:
- `thirdparty/phantom-fhe/src/rns_bconv.cu`의 `DRNSTool::modup`
- `thirdparty/phantom-fhe/src/rns_bconv.cu`의 `DRNSTool::moddown_from_NTT`
- `thirdparty/phantom-fhe/src/eval_key_switch.cu`의 `keyswitch_inplace`

### Phantom의 \(\alpha,\beta\) 정의

- \(\alpha = |P| =\) `special_modulus_size`
- \(\beta = \lceil |Ql| / \alpha \rceil\)

참고 코드:
- `thirdparty/phantom-fhe/src/rns.cu`에서 `beta = ceil(size_Ql / alpha)` 및 part-Ql을 \(\alpha\)개씩 분할.

### (핵심) Phantom CKKS에서 limb-적용량 기준 NTT/INTT 총량

기호:
- \(T := |QlP| = |Ql| + |P|\)
- \(|P| = \alpha\)
- \(|Ql| = T-\alpha\)

**ModUp**
- **INTT**: \(|Ql| = T-\alpha\) (입력이 NTT라서 Ql을 normal로 내림)
- **NTT**: digit마다 QlP 전체를 fwd하지 않고, digit의 part-Ql 구간은 이미 NTT이므로 제외한다.
  - digit \(i\)의 제외량 = `size_PartQl(i)`
  - 합 제외량 = \(\sum_i \text{size\_PartQl}(i) = |Ql|\)
  - 따라서 modup NTT 총량:
    \[
    \sum_i (T-\text{size\_PartQl}(i)) = \beta T - |Ql| = \beta T - (T-\alpha)
    \]

**ModDown (×2)**
- CKKS는 **P만 INTT**로 내린다 → **INTT 총량 = \(2|P| = 2\alpha\)**
- 마지막 단계는 Ql에서 NTT 도메인으로(fused) 처리 → **NTT 총량 = \(2|Ql| = 2(T-\alpha)\)**

따라서 “fwd 합”과 “inv 합”으로 보면:
- fwd 총량 \(=\;(\beta T-(T-\alpha)) + 2(T-\alpha) = \beta T + (T-\alpha)\)
- inv 총량 \(=\;(T-\alpha) + 2\alpha = T+\alpha\)

---

## 시뮬레이터 정합 (Phantom 최적화 반영)

`EngineModel::enqueue_keyswitch_phantom_ckks`를 Phantom(CKKS)와 동일한 NTT/INTT limb-적용량이 나오도록 수정했다:

- **modup fwd NTT**: digit별로 `T - size_PartQl`만 enqueue하도록 변경 (Phantom의 `exclude_range` 반영)
- **moddown inverse NTT**: `QlP` 전체가 아니라 **`P-only`**만 enqueue하도록 변경
- **moddown forward (fused)**: `Ql`에 대해 forward NTT를 **2회** 추가 (Phantom `nwt_..._forward_inplace_fuse_moddown` 모델링)

동시에 analytic mirror(`keyswitch_op_profile.h`)의 `weighted_ntt_fwd_coeff_elems`/`weighted_ntt_inv_coeff_elems`도 위 수식으로 갱신했다.

검증:
- `alpha=1..4`에서 `weighted_ntt_coeff_elems`와 `eng_ntt_coeff_ops`의 diff가 **0.0**으로 일치함.

관련 파일:
- `src/include/source/sim/engine_model.h`
- `src/include/source/sim/keyswitch_op_profile.h`

---

## 실행 방법 (재현)

### CSV 생성 + 플롯(권장 스크립트)

```bash
cd /home/jyg/projects/MOAI_GPU
src/scripts/run_hybrid_ks_sweep_plots.sh
```

기본 env:
- `MOAI_SIM_HYBRID_KS_ALPHA_RANGE=1-35`
- `MOAI_SIM_HYBRID_KS_EXACT_PARTITION=1` (나눠지지 않으면 제외)
- `MOAI_SIM_BACKEND=1`
- `MOAI_SIM_HYBRID_KS_MEASURE_ENGINE=1`

모든 α 포함하려면:

```bash
MOAI_SIM_HYBRID_KS_EXACT_PARTITION=0 src/scripts/run_hybrid_ks_sweep_plots.sh
```

### 직접 플롯 실행

```bash
python3 src/scripts/plot_hybrid_ks_profile.py --all-plots --csv output/sim/hybrid_ks_profile.csv
```

### (추가) α sweep에 따른 modulus size 텍스트/그래프 출력 (T 고정, BTS 고정)

전제(예: 동일 security bit 유지 목적):
- `T=36` 고정
- `BTS=14` 고정
- `SPECIAL=α`
- `GENERAL = T - BTS - α`

텍스트 표(직관적인 숫자 출력) + 파일 저장:

```bash
python3 src/scripts/plot_hybrid_ks_profile.py --mod-sizes-txt --T 36 --bts 14 --csv output/sim/hybrid_ks_profile.csv
```

스택 바 PNG:

```bash
python3 src/scripts/plot_hybrid_ks_profile.py --mod-sizes --T 36 --bts 14 --csv output/sim/hybrid_ks_profile.csv
```

주요 출력물:
- `output/sim/hybrid_ks_profile.csv`
- `output/sim/hybrid_ks_profile_stacked.png`
- `output/sim/hybrid_ks_profile_pie.png` (도넛 %)
- `output/sim/hybrid_ks_profile_engine_compare.png`
- `output/sim/hybrid_ks_profile_stacked_cycles.png`
- `output/sim/hybrid_ks_profile_engine_cycles.png`
- `output/sim/hybrid_ks_profile_memory.png`
- `output/sim/hybrid_ks_profile_mod_sizes.txt` (α→GENERAL/BTS/SPECIAL 표)
- `output/sim/hybrid_ks_profile_mod_sizes.png` (α→GENERAL/BTS/SPECIAL 스택 바)

---

## 새 채팅에 붙여넣을 프롬프트(작업 이어가기)

아래를 새 채팅 첫 메시지로 그대로 붙여넣으면 됨.

```text
MOAI_GPU에서 hybrid keyswitch profile 정리/검증 작업을 이어가고 싶다.

현재까지 변경:
- run_hybrid_ks_sweep_plots.sh: MOAI_SIM_HYBRID_KS_EXACT_PARTITION 기본=1 (|Ql|≥α and |Ql| mod α≠0 스킵). 0이면 전 α 포함.
- engine_model.h: VEC 통계를 vec_bconv(키스위치 modup/moddown bconv) vs vec_arith(mul/add)로 분리.
- keyswitch_op_profile.h: CSV에 eng_vec_bconv_* / eng_vec_arith_* (enq, coeff_ops, busy_cyc) 추가.
- plot_hybrid_ks_profile.py: 스택 5단(NTT fwd/INTT/BConv/vec_mul/vec_add), 엔진 비교 2×2, cycles split, pie(%) 출력 추가.
- Phantom CKKS 정합: EngineModel의 enqueue_keyswitch_phantom_ckks 및 analytic weighted 식을 Phantom modup/moddown_from_NTT 구현(exclude-range, P-only INTT, Ql-only fused forward)에 맞게 수정했고 alpha=1..4에서 analytic vs engine NTT coeff_ops diff=0 확인함.

다음으로 하고 싶은 일:
- T=36 고정, BTS=14 고정(동일 hardware security bit 유지) 가정에서 α sweep에 따라 `SPECIAL=α`, `GENERAL=T-BTS-α`가 변하는 걸 시각화(그래프). `plot_hybrid_ks_profile.py --mod-sizes`로 `GENERAL/BTS/SPECIAL` 스택 바 PNG 추가 출력.

참고 문서: md/hybrid_ks_profile_worklog.md
```

