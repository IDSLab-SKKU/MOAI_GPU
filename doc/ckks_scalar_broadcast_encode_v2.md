# CKKS scalar broadcast encode 최적화 (v1→v2 정리)

이 문서는 MOAI/Phantom(CUDA)에서 **브로드캐스트 스칼라 평문(전 슬롯 동일 실수)**을 `ct×pt`에서 반복 사용할 때,
CKKS 인코딩/평문 곱(`multiply_plain`) 경로를 단계적으로 최적화한 변경 내용을 정리한다.

작성일: 2026-04-17  
레포: `MOAI_GPU/thirdparty/phantom-fhe` + `MOAI_GPU/src/*`

---

## 목표

- `encoder.encode(double, ...)` 형태의 **스칼라 브로드캐스트**가중치 \(w\)에 대해
  - **IFFT(슬롯 임베딩)** 제거
  - **plain forward NTT** 제거
  - (확장) **plain 전체 버퍼(materialization)** 제거
- 기능 동치:
  - **평문 버퍼(uint64 RNS/NTT)** bitwise 동치
  - `multiply_plain` 후 **복호화(decrypt/decode) 결과** 동치
- 프로파일에서 IFFT/NTT 관련 커널이 사라지거나 대체되는지 확인

---

## 배경: 기존 Phantom CKKS 인코딩 흐름

일반 벡터 인코딩(슬롯 벡터 입력)은 Phantom에서 대략 아래 단계를 탄다.

1) 슬롯 벡터 → `special_fft_backward` (IFFT 계열)  
2) `decompose_array` (복소 버퍼를 정수화/RNS 분해)  
3) `nwt_2d_radix8_forward_inplace` (plain forward NTT)  

`multiply_plain`(CKKS)은 ciphertext가 NTT form이라고 가정하고, plain도 NTT form으로 준비된 `PhantomPlaintext.data()`를
`multiply_rns_poly`로 pointwise 곱한다.

---

## 핵심 관찰: 스칼라 브로드캐스트의 구조

슬롯에 동일 실수 \(w\)를 브로드캐스트할 때, Phantom의 구현/레이아웃에서 다음이 성립함을 실제로 확인했다.

- 스칼라(uniform) 경로의 `decompose_array` 직후(즉, forward NTT 전) 계수 도메인은
  - 각 modulus limb \(q_j\)에 대해 **index 0만 비영**, 나머지 계수는 0
- 즉 계수 다항식은 사실상 **상수항만 가진 다항식**
- 이 경우 NTT/eval 도메인에서는 limb별로 상수 브로드캐스트 형태로 동치:
  - limb \(j\): \([c \bmod q_j,\; c \bmod q_j,\; ...]\)
  - 여기서 \(c = \mathrm{round}(w\cdot scale)\)

---

## 변경 요약

### (A) v0 → v1: IFFT 스킵 + NTT를 “브로드캐스트 fill”로 대체

- 대상: `PhantomCKKSEncoder::encode_internal_uniform_real`
- IFFT를 아예 호출하지 않고, `decompose_array` 입력 복소 버퍼를 DC 한 점으로 구성:
  - `gpu_ckks_msg_vec_->in()[0] = (w*scale, 0)` 나머지 0
- `decompose_array` 이후에는 forward NTT 대신
  - 각 limb 블록에서 index0 값을 읽어 limb 전체로 복제하는 CUDA 커널로 대체

이 단계에서 `ct_pt` 프로파일은 `ckks_broadcast_slot0_per_modulus_inplace`가 뜨고
`inplace_special_ifft_*`가 사라지는 것을 확인했다.

### (B) v1 → v2: plain materialization 제거 + multiply_plain fast-path

v1은 still “plain full buffer”를 만들었다(브로드캐스트 fill이긴 하지만 \(N\times L\) 전체).
사용 요구사항은:

- **H2D는 원본 스칼라만 이동**
- GPU에서 limb 분해(=각 \(q_j\)에서 mod)하고, NTT/버퍼 생성은 스킵

따라서 v2에서는:

- `PhantomPlaintext`에 broadcast-plain 모드를 추가하고,
  - per-limb 배열(스칼라 limb 튜플)을 저장하지 않음
  - 오직 `broadcast_scalar_coeff = llround(w*scale)` 하나만 저장
- `multiply_plain` / `add_plain` / `sub_plain`에서 broadcast-plain이면
  - 커널에서 **per-thread limb modulus \(q_j\)**를 보고 `coeff mod qj`를 즉석 계산하여 사용
  - plain full buffer 없이 곱/덧/뺄 수행

---

## 수정된 파일 목록

### Phantom / thirdparty

- `MOAI_GPU/thirdparty/phantom-fhe/src/ckks.cu`
  - `PhantomCKKSEncoder::encode_internal_uniform_real`:
    - IFFT 제거(DC only) 유지
    - v2에서 `broadcast_scalar_coeff_ = llround(w*scale)`만 저장하도록 변경
  - `#include "util/uintarithsmallmod.h"` 추가 (compute_shoup 관련 include 정리용)

- `MOAI_GPU/thirdparty/phantom-fhe/include/plaintext.h`
  - broadcast-plain 표현 추가
    - `rep_` (`full_ntt` vs `broadcast_ntt`)
    - `int64_t broadcast_scalar_coeff_` (스칼라 1개)
    - `is_ckks_broadcast_ntt()` / `broadcast_scalar_coeff()` API

- `MOAI_GPU/thirdparty/phantom-fhe/src/evaluate.cu`
  - CKKS:
    - `multiply_plain_ntt`에 broadcast-plain 분기 추가:
      - 새 커널 `multiply_broadcast_scalar_coeff_rns_poly_inplace`
    - `add_plain_inplace` / `sub_plain_inplace`에 broadcast-plain 분기:
      - 새 커널 `add_broadcast_scalar_rns_poly_inplace`
      - 새 커널 `sub_broadcast_scalar_rns_poly_inplace`

### MOAI 테스트/벤치

- `MOAI_GPU/src/include/test/matrix_mul/test_ct_pt_matrix_mul.cuh`
  - `ct_pt_matrix_mul_sanity_small_test()` 추가 (poly_degree=4096)
  - `MOAI_ENC_EQ_CHECK=1` 비교에서, broadcast-plain인 경우
    - `broadcast_scalar_coeff`와 modulus로 per-limb 브로드캐스트 값을 재구성해 legacy vec와 비교

- `MOAI_GPU/src/test.cu`
  - `MOAI_BENCH_MODE=ct_pt_sanity_small` 분기 추가

---

## 검증 방법 & 결과

### 1) 복호화 기반 sanity (원문 기대값 비교)

- `MOAI_BENCH_MODE=ct_pt_sanity`
  - legacy: `encoder.encode(wvec(slot_count,w))`
  - fast: `encoder.encode(w)`
  - `multiply_plain` 후 decrypt/decode
  - 기대값 \(x[s]\cdot w\) 및 fast-vs-legacy diff 확인 → PASS

- `MOAI_BENCH_MODE=ct_pt_sanity_small` (새로 추가)
  - poly_degree=4096에서도 동일 sanity 수행 → PASS

### 1-b) sanity 확장: add/sub 포함 + 예외 케이스(안전성)

위 sanity를 `multiply_plain`뿐 아니라 `add_plain_inplace`, `sub_plain_inplace`까지 확장하여
fast(uniform scalar) vs legacy(vector encode) 경로가 모두 동치임을 확인했다.

또한 아래 “잘못된 사용”은 조용히 진행하지 않고 예외로 reject되도록 보강하고, sanity에서 throw를 확인했다.

- NaN/Inf 입력: `encoder.encode(NaN/Inf, ...)` → throw
- overflow 입력: `llround(value*scale)`가 `int64`에 못 들어가는 케이스 → throw
- scale mismatch: ciphertext/ plaintext scale 불일치 → throw
- chain mismatch: ciphertext/ plaintext chain_index 불일치 → throw

### 2) 평문 버퍼 bitwise 동치 체크

- `MOAI_BENCH_MODE=ct_pt MOAI_ENC_EQ_CHECK=1`
  - legacy vec encode vs scalar encode 결과의 `PhantomPlaintext`를 uint64 배열로 비교
  - broadcast-plain일 때는 테스트에서 전개(reconstruct)하여 비교
  - `diff_cnt=0` 확인

---

## 프로파일(CT×PT micro) 결과 포인트

프로파일 스크립트:

- `MOAI_GPU/src/scripts/profile_ct_pt_micro.sh`
  - NVTX 필터: `moai:ct_pt_matrix_mul_wo_pre`

v2 scalar-only 구현에서는 NVTX clean TSV에서:

- `multiply_broadcast_scalar_coeff_rns_poly_inplace`가 지배적으로 나타남
- `decompose_array_uint64`, `special_ifft`, plain forward NTT 관련 커널이 나타나지 않음(해당 슬라이스 내)

예시 산출물:

- `MOAI_GPU/output/ct_pt/ct_pt_micro_v2_scalar_only_run_run1.nsys-rep`
- `MOAI_GPU/output/ct_pt/ct_pt_micro_v2_scalar_only_run_run1.kern_sum.mul_nvtx_clean.tsv`

---

## 주의/제약

- v2는 “스칼라 브로드캐스트 plain”에만 적용된다.
  - LayerNorm의 gamma/beta처럼 bias mask가 들어간 **sparse mask 벡터**는 별도 최적화 필요
- `broadcast_scalar_coeff_ = llround(w*scale)`는 표현 범위/특이값 이슈가 있으므로
  - NaN/Inf 입력 및 `int64` overflow 케이스를 **encode 단계에서 reject(throw)** 하도록 보강함
  - 커널 측에서도 `INT64_MIN` 관련 UB가 없도록 abs 계산을 안전한 형태로 처리함
- 현재 구현은 “GPU에서 decompose_array를 호출”하는 형태가 아니라,
  - GPU 연산에서 modulus별 mod를 즉석 계산하는 방식으로 decompose/NTT를 우회한다.

---

## 재현 커맨드 모음

빌드:

```bash
cmake --build /home/jyg/projects/MOAI_GPU/build --target test -j8
```

sanity:

```bash
cd /home/jyg/projects/MOAI_GPU/build
MOAI_BENCH_MODE=ct_pt_sanity ./test
MOAI_BENCH_MODE=ct_pt_sanity_small ./test
```

평문 bitwise 동치:

```bash
cd /home/jyg/projects/MOAI_GPU/build
MOAI_BENCH_MODE=ct_pt MOAI_ENC_EQ_CHECK=1 ./test
```

프로파일:

```bash
cd /home/jyg/projects/MOAI_GPU
MOAI_PROFILE_OUT=ct_pt_micro_v2_scalar_only_run \
MOAI_PROFILE_DIR=/home/jyg/projects/MOAI_GPU/output/ct_pt \
MOAI_NSYS_TRACE=cuda,nvtx \
MOAI_NSYS_STATS_FILTER_NVTX=1 \
MOAI_KERN_EXPORT=1 \
timeout 180s bash src/scripts/profile_ct_pt_micro.sh
```

