# CKKS 상수 슬롯 인코딩 fast path — 구현·프로파일 핸드오프

새 채팅에 그대로 붙여 넣을 수 있는 요약입니다. 배경은 `MOAI_GPU/doc/micro_profile_operations_index.md` §6.A (`ct_pt_matrix_mul_wo_pre`가 스칼라 `W[j][i]`마다 `Encoder::encode(double,…)`로 전 슬롯 벡터 + `encode_internal`의 `special_fft_backward`를 매번 탐).

---

## 1) 문제(기존 경로)

- `moai::Encoder::encode(double, …)` → `vector<double>(slot_count, value)` 생성 후 `PhantomCKKSEncoder::encode` → `encode_internal`.
- `encode_internal` (`thirdparty/phantom-fhe/src/ckks.cu`)은
  - H2D로 슬롯 길이 복소 버퍼 복사
  - `bit_reverse_and_zero_padding`
  - **`special_fft_backward`** (프로파일에서 `inplace_special_ifft_*`)
  - `decompose_array` → **`nwt_2d_radix8_forward_inplace`**
- 상수 브로드캐스트(모든 슬롯 동일 실수)에도 위 IFFT 계열이 **매 encode마다** 반복됨.

---

## 2) 아이디어(수학)

슬롯 입력이 전부 동일한 실수 \(w\)인 경우, `bit_reverse_and_zero_padding` 이후 GPU 버퍼는 “모든 슬롯에 \((w,0)\)” 형태이고, `special_fft_backward`는 **마지막에 `fix = scale / sparse_slots`를 곱하는 선형 연산**이 핵심이다.

따라서 (실수 스케일에 대해) 대략

\[
\text{IFFT\_path}(w\cdot \mathbf{1}) \;=\; w \cdot \text{IFFT\_path}(\mathbf{1})
\]

처럼 취급할 수 있어, **`L(\mathbf{1})`만 한 번** `special_fft_backward(..., scalar=1.0)`로 만들어 GPU에 두고, 이후 각 스칼라는

- `out = basis * (w * fix)`

로 치환 가능. 이후 `decompose_array` + `nwt_2d_radix8_forward_inplace`는 기존과 동일.

주의: “최종 다항식 계수가 전부 같다”는 의미가 아니라, **브로드캐스트 슬롯 메시지의 인코딩 결과가 선형변환으로 결정**된다는 뜻.

---

## 3) 코드 변경(파일)

### A) Phantom: uniform fast path

- `thirdparty/phantom-fhe/include/ckks.h`
  - `encode_internal_uniform_real(...)` (private)
  - `encode_uniform_real(...)` (public inline): `destination.resize(...)` 후 uniform 경로 호출

- `thirdparty/phantom-fhe/src/ckks.cu`
  - 전역(익명 namespace) 캐시:
    - `std::mutex` + `std::unordered_map<uint32_t, CkksUniformSlotBasisCache>` 키: **`slots`**
    - `get_ckks_uniform_basis(slots, gp, stream)`:
      - `gp.set_sparse_slots(slots)`
      - `ckks_fill_unit_real_kernel`로 `gp.in()`을 전 슬롯 `(1,0)`로 채움
      - `special_fft_backward(gp, 1.0, stream)`
      - 결과를 `basis` GPU 버퍼로 D2D 복사
      - 호스트로 한 번 복사해 `max_abs` 계산(이후 `max_coeff_bit_count` 추정)
  - `PhantomCKKSEncoder::encode_internal_uniform_real(...)`:
    - `sparse_slots_ = slots_` 강제(풀 브로드캐스트와 동일하게 맞춤)
    - `ckks_scale_complex_by_real_kernel`로 `basis * (value * scale/slots)`를 `gp.in()`에 생성
    - `decompose_array` / `nwt_2d_radix8_forward_inplace`는 기존과 동일

### B) MOAI: double encode가 uniform 경로를 타도록 연결

- `src/include/source/ckks_evaluator_parallel.cuh`
  - `Encoder::encode(double, chain_index, scale, ...)` 및 `encode(double, scale, ...)`가
    `encoder->encode_uniform_real(...)` 호출로 변경 (호스트에서 `vector(slot_count,value)` 생성 제거)

---

## 4) 프로파일 검증(결과 요약)

스크립트: `src/scripts/profile_ct_pt_micro.sh`  
NVTX 필터: `moai:ct_pt_matrix_mul_wo_pre`

관측(`output/ct_pt/ct_pt_micro_run1.kern_sum.mul_nvtx_clean.tsv`):

- 기존 병목이던 `inplace_special_ifft_base_kernel` / `iter_kernel`이 **사실상 제거**(캐시 워밍 수준의 극소 호출만 남음)
- 새로 `ckks_scale_complex_by_real_kernel`가 소량 비중으로 등장 (스칼라 스케일)
- 상위 비중은 `decompose_array_uint64` + `inplace_fnwt_radix8_*` + `multiply_rns_poly` 등 **RNS/NTT/곱** 쪽으로 이동 (예상된 다음 병목)

차트가 자동 생성되지 않으면(환경에 따라 `export` 단계가 조용히 실패할 수 있음):

```bash
cd ~/projects/MOAI_GPU
python3 src/scripts/export_ct_pt_kern_sum.py \
  output/ct_pt/ct_pt_micro_run1.kern_sum.tsv \
  output/ct_pt/ct_pt_micro_run1.kern_sum.mul_nvtx.tsv
```

---

## 5) 빌드/환경 이슈(재현 메모)

- `test` 타깃은 부트스트랩 소스 때문에 **NTL 헤더/라이브러리**가 필요.
- `CMakeLists.txt`는 `CONDA_PREFIX/include`를 include path로 넣을 수 있는데, **`build/`가 예전 conda prefix로 configure 되면** `includes_CUDA.rsp`에 잘못된 `/home/.../anaconda3/include`가 박혀 NTL을 못 찾는 현상이 생김.
  - 해결: `moai_gpu` env에서 **`build/` 재configure**(캐시 삭제 또는 build 폴더 재생성).

---
*작성 위치: `MOAI_GPU/doc/ckks_uniform_encode_fast_path_handoff.md`*
