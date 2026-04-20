# CKKS scalar broadcast v2 — 작업 요약 (Git 제외)

새 채팅에서 바로 이어서 작업할 수 있도록, **Git 히스토리/푸시 이슈를 제외하고** 이번에 적용/검증/벤치/플로팅까지 진행한 내용을 한 곳에 정리한다.

작성일: 2026-04-17  
범위: `MOAI_GPU/thirdparty/phantom-fhe` + `MOAI_GPU/src/*`

---

## 1) 목표 / 결과 한줄 요약

- **목표**: CKKS에서 `encoder.encode(double w, ...)` 형태의 **스칼라 브로드캐스트 평문**(모든 슬롯 동일)을 CT×PT에서 반복 사용할 때, legacy(슬롯 벡터 encode + IFFT/NTT) 비용을 제거/축소.
- **결과**: v2에서 스칼라 브로드캐스트 평문을 **full plain buffer 없이** 표현하고, `multiply_plain`/`add_plain_inplace`/`sub_plain_inplace`가 **broadcast-plain fast-path**로 동작.

관련 상세 설계/구현 설명은 기존 문서 참고:
- `doc/ckks_scalar_broadcast_encode_v2.md`

---

## 2) 구현 핵심 (v2 스칼라 브로드캐스트 평문 표현)

- **Plaintext 표현**: `PhantomPlaintext`에 `broadcast_ntt` representation 추가
  - `broadcast_scalar_coeff_ = llround(w * scale)` 단 하나만 저장
  - 커널에서 limb modulus \(q_j\)를 보고 `coeff mod qj`를 즉석 계산하여 사용
- **연산 fast-path**:
  - `multiply_plain` → `multiply_broadcast_scalar_coeff_rns_poly_inplace`
  - `add_plain_inplace` → `add_broadcast_scalar_rns_poly_inplace`
  - `sub_plain_inplace` → `sub_broadcast_scalar_rns_poly_inplace`

---

## 3) 안정성/오류 체크(encode 및 연산 경로)

### 3.1 encode 단계 reject(throw)

스칼라 encode(`encode_uniform_real`)에서 아래 케이스를 조용히 진행하지 않고 예외로 처리:

- NaN/Inf 입력
- `llround(w*scale)`가 `int64` 범위를 넘어가는 overflow
- `llround` 결과가 `INT64_MIN`이 되는 케이스(커널에서 abs 처리 UB 방지와 함께 방어)

### 3.2 evaluator 연산 단계 mismatch reject(throw)

`multiply_plain`, `add_plain_inplace`, `sub_plain_inplace`에서:

- **scale mismatch**: ciphertext/ plaintext scale 불일치
- **chain mismatch**: ciphertext/ plaintext chain_index 불일치

---

## 4) “진짜 legacy(IFFT 포함)” vs v2 비교 스위치

MOAI wrapper(`moai::Encoder`)에서 스칼라 입력을 강제로 **슬롯 벡터 encode**로 보내 “진짜 legacy(IFFT/NTT 포함)” 비용을 타게 할 수 있음.

- **환경변수**: `MOAI_SCALAR_ENCODE_LEGACY_VEC=1`
- 적용 위치: `src/include/source/ckks_evaluator_parallel.cuh` 의 `Encoder::encode(double, ...)`

즉,
- `MOAI_SCALAR_ENCODE_LEGACY_VEC=1` → legacy(슬롯벡터) encode 경로
- unset(기본) → v2 fast-path(`encode_uniform_real`) 경로

---

## 5) 검증/테스트(추가 sanity 포함)

### 5.1 복호화 기반 sanity

다음 모드에서 fast(스칼라) vs legacy(벡터) 경로 동치성을 확인:

- `MOAI_BENCH_MODE=ct_pt_sanity`
- `MOAI_BENCH_MODE=ct_pt_sanity_small` (poly_degree=4096)

기본적으로 `multiply_plain` 뿐 아니라 `add_plain_inplace`, `sub_plain_inplace`까지 포함해 sanity를 확장했고,
NaN/Inf/overflow/mismatch 케이스에 대해 throw도 확인하도록 보강됨.

### 5.2 평문 bitwise 동치 체크(재구성 비교)

- `MOAI_BENCH_MODE=ct_pt MOAI_ENC_EQ_CHECK=1`
- broadcast-plain은 테스트에서 `broadcast_scalar_coeff` + modulus들로 limb별 브로드캐스트 배열을 재구성해 legacy vec 결과와 비교.

---

## 6) 단일레이어 설정 기반 CTxPT “Projection별” 벤치 (QKV/out/FC1/FC2)

“실제 single-layer inference 설정과 동일 조건”에서, `qkv/out/fc1/fc2`를 **각각 별도 프로세스**로 실행하여 OOM 안전하게 비교.

### 6.1 실행 스크립트

- `src/scripts/bench_ct_pt_proj_compare.sh`
  - 각 op를 `./test` 별도 실행
  - 결과 저장: `output/ct_pt_proj_compare/run_<op>_<mode>.txt` + `summary.json`

주요 환경변수:

- `MOAI_CT_PT_PROJ_MODE=legacy|v2|both` (default: `both`)
- `MOAI_CT_PT_PROJ_OPS="qkv,out,fc1,fc2"` (default: all)
- `MOAI_CT_PT_PROJ_OP=<op>`: (스크립트 내부에서 설정)
- `MOAI_BENCH_MODE=ct_pt_proj_compare`: (스크립트 내부에서 설정)
- `MOAI_TEST_OUTPUT_DISABLE=1`: (스크립트 내부에서 설정; stdout redirect 방지해서 `tee` 로그 파싱 가능)

### 6.2 시간 출력 키(파서가 보는 라인)

벤치 출력은 다음 패턴을 사용(스크립트의 파서가 이를 읽어 `summary.json` 생성):

- `[CT_PT_PROJ] QKV_proj time_s=...`
- `[CT_PT_PROJ] out_proj time_s=...`
- `[CT_PT_PROJ] fc1 time_s=...`
- FC2는 compute만 따로 뽑기 위해:
  - `[CT_PT_PROJ] fc2_precompute time_s=...`
  - `[CT_PT_PROJ] fc2_compute time_s=...`

---

## 7) FC2 OOM 대응: VRAM residency 모드 + precompute/compute 분리

FC2는 입력 CT 수가 매우 커서(예: 3072) “입력 CT 전부를 GPU에 상주시켜서” 돌리면 OOM이 날 수 있음.
이를 위해 FC2에 **입력 CT의 VRAM 상주 방식**을 선택하는 모드를 추가했고, 측정을 **precompute vs compute**로 분리함.

### 7.1 모드 선택 환경변수

- `MOAI_CT_PT_FC2_MODE=full_vram|chunk_vram|stream` (default: `chunk_vram`)
  - **full_vram**: 입력 CT를 GPU에 전부 올린 채 compute (원래 “전부 VRAM” 방식에 가장 가까움)
  - **chunk_vram**: `chunk_size` CT만 GPU에 올려서 여러 번 나눠 compute
  - **stream**: 1 CT씩만 올려 compute (chunk_size=1)
- `MOAI_CT_PT_FC2_CHUNK=<n>`: `chunk_vram`일 때 chunk 크기(1..3072 범위에서 사용)

### 7.2 precompute vs compute 의미

- **precompute**: “입력은 암호문” semantics를 유지하기 위해, FC2 입력 CT들을 **host 쪽에 준비**(암호화/준비 단계 포함)하는 비용
- **compute**: host에 준비된 CT들을 GPU로 (full/chunk/stream 방식으로) 옮겨 실제 CT×PT matmul을 수행하는 비용

그래프/비교에는 보통 **compute(=핵심 연산)**를 사용하고, 필요하면 precompute도 함께 보고 병목을 분리함.

---

## 8) 결과 요약/플로팅(Projection 벤치)

- `output/ct_pt_proj_compare/summary.json`을 읽어 그림 생성
- 스크립트: `src/scripts/plot_ct_pt_proj_compare.py`

기본값은 “amortized per batch” (num_X=256으로 나눔)이며, 표시 단위는 ms.

옵션:
- `--mode raw|amortized_batch|amortized_token|amortized_batch_token` (default: `amortized_batch`)
- `--num-x`, `--num-row`로 summary.json meta를 override 가능

출력:
- `output/ct_pt_proj_compare/legacy_only.png`
- `output/ct_pt_proj_compare/v2_only.png`
- `output/ct_pt_proj_compare/compare.png` (둘 다 있을 때)

---

## 9) 빠른 재현 커맨드

### 9.1 빌드

```bash
cmake --build /home/jyg/projects/MOAI_GPU/build --target test -j"$(nproc)"
```

### 9.2 sanity (추가 검증)

```bash
cd /home/jyg/projects/MOAI_GPU/build
MOAI_BENCH_MODE=ct_pt_sanity ./test
MOAI_BENCH_MODE=ct_pt_sanity_small ./test
```

### 9.3 Projection 벤치(각 op 별도 프로세스)

```bash
cd /home/jyg/projects/MOAI_GPU

# v2만
MOAI_CT_PT_PROJ_MODE=v2 bash src/scripts/bench_ct_pt_proj_compare.sh

# legacy만(진짜 IFFT/NTT 포함 legacy를 타게 함)
MOAI_CT_PT_PROJ_MODE=legacy bash src/scripts/bench_ct_pt_proj_compare.sh

# 둘 다 비교
MOAI_CT_PT_PROJ_MODE=both bash src/scripts/bench_ct_pt_proj_compare.sh
```

FC2 VRAM 모드 예시:

```bash
# 기존에 가깝게(전부 VRAM 상주) — OOM 가능성 있음
MOAI_CT_PT_PROJ_MODE=v2 MOAI_CT_PT_PROJ_OPS="fc2" MOAI_CT_PT_FC2_MODE=full_vram \
  bash src/scripts/bench_ct_pt_proj_compare.sh

# chunk 크기 지정
MOAI_CT_PT_PROJ_MODE=v2 MOAI_CT_PT_PROJ_OPS="fc2" MOAI_CT_PT_FC2_MODE=chunk_vram MOAI_CT_PT_FC2_CHUNK=256 \
  bash src/scripts/bench_ct_pt_proj_compare.sh

# 1개씩 스트리밍
MOAI_CT_PT_PROJ_MODE=v2 MOAI_CT_PT_PROJ_OPS="fc2" MOAI_CT_PT_FC2_MODE=stream \
  bash src/scripts/bench_ct_pt_proj_compare.sh
```

### 9.4 플롯

```bash
cd /home/jyg/projects/MOAI_GPU
python3 src/scripts/plot_ct_pt_proj_compare.py --mode amortized_batch
```

