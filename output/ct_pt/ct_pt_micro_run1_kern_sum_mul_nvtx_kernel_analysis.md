# `ct_pt_micro_run1.kern_sum.mul_nvtx_clean.tsv` 커널 분석

이 문서는 Nsight Systems에서 NVTX 범위 **`moai:ct_pt_matrix_mul_wo_pre`** 로 잘린 GPU 커널 합계(`kern_sum.mul_nvtx`)에 나타난 커널을, **Phantom-FHE** CUDA 구현 기준으로 정리한 것이다. 수치·비율은 해당 프로파일 한 번의 결과이며, 환경에 따라 달라질 수 있다.

## 프로파일 맥락

- **애플리케이션**: MOAI `ct_pt_matrix_mul_wo_pre` — 암호문 열과 평문(실수) 웨이트 행렬 곱 과정에서, 웨이트를 슬롯 단위로 **CKKS 인코딩**하고 **평문–암호문 곱셈** 등이 수행된다.
- **라이브러리**: `thirdparty/phantom-fhe` — CKKS `encode_internal`의 `special_fft_backward`, `decompose_array`, `nwt_2d_radix8_forward_inplace` 등과, 다항식 연산용 **RNS(잔여 수 체계)** 커널이 여기에 대응한다.

아래 표는 `*_clean.tsv`의 **ShortName** 순서(비중 큰 순)에 맞춘 요약이다.

| ShortName | 대략적 역할 |
|-----------|-------------|
| `inplace_special_ifft_base_kernel` | CKKS 인코딩: 슬롯 도메인 → 계수 도메인으로 가는 **역 FFT 계열(백워드 butterfly)** 의 한 단계(큰 `n`에서 공유 메모리 기반 베이스 스테이지) |
| `inplace_special_ifft_iter_kernel` | 동일 백워드 변환의 **반복 스테이지**(큰 차수에서 여러 butterfly 라운드) |
| `inplace_fnwt_radix8_phase1` / `phase2` | **순방향 NTT**(모듈러스별 `q_i`에서의 negacyclic NTT) — radix-8 2단계 파이프라인 |
| `multiply_rns_poly` | RNS 다항식 **성분별 곱셈** mod `q_i` (NTT 도메인 또는 계수 도메인에서의 plain–ct 곱의 핵심) |
| `add_rns_poly` | RNS 다항식 **덧셈** mod `q_i` (중간 결과 합성, rescale/keyswitch 조각 등에 사용) |
| `phantom::arith::decompose_array_uint64` | 복소(실수) 계수를 **라운딩 후 각 소모듈러스 `q_i`로 투영**해 `uint64` RNS 계수로 펼침 |
| `bit_reverse_and_zero_padding` | 인코딩 전: 입력 슬롯을 **비트 리버스 순서**로 놓고, 스파스 슬롯 크기에 맞게 **0 패딩** |
| `inplace_inwt_radix8_phase1` / `phase2` | **역 NTT**(INTT) radix-8 2단계 — 계수/NTT 표현을 바꿀 때(예: 마지막 모듈러스 제거 전 **INTT** 등) |
| `divide_and_round_ntt_inv_scalar_kernel` | **마지막 모듈러스 `q_L` 제거(divide & round)** 파이프라인에서, NTT 도메인에서 차이에 `q_L^{-1}` 곱 등 |
| `divide_and_round_reduce_q_last_kernel` | 위와 같은 **divide & round** 단계에서 `q_L` 타워 계수를 다른 `q_j`에 맞게 **축약·반올림** |

---

## 1. `bit_reverse_and_zero_padding`

**파일**: `thirdparty/phantom-fhe/src/ckks.cu`

**역할**

- 호스트에서 넘어온 길이 `values_size`의 슬롯 값(복소 배열)을 GPU 버퍼에 올린 뒤, **차수 `slots = 2^{logn}`** 짜리 작업 버퍼에 쓴다.
- 각 인덱스 `tid`에 대해 `dst[reverse_bits(tid)] = src[tid]` 형태로 **비트 리버스 순열**을 적용한다. 이는 이후 **special FFT / IFFT**가 요구하는 데이터 순서를 맞추기 위한 전형적인 CKKS/SEAL 계열 패킹이다.
- `tid >= in_size`인 위치는 **0으로 채워** 실제 메시지 길이보다 작은 경우 스파스 슬롯 상한까지 확장한다.

**이 구간에서의 의미**: `wo_pre` 경로에서 **매 웨이트 스칼라(또는 벡터) 인코딩**마다 호출될 수 있어, 호출 빈도가 높다.

---

## 2. `inplace_special_ifft_base_kernel` / `inplace_special_ifft_iter_kernel`

**파일**: `thirdparty/phantom-fhe/src/fft.cu` — `special_fft_backward()`

**역할**

- CKKS **인코딩**의 핵심: 슬롯에 깔린 값(복소 벡터)을 **“메시지 다항식” 계수 쪽 표현**으로 옮기기 위한 **역 Cooley–Tukey형 butterfly** 연산이다. 주석상 *backward NTT transformation* 으로 적혀 있으나, 연산체는 **복소수 `double2`**, twiddle은 `2n`차 원시근 관련 테이블로, **정수 NTT가 아니라 인코딩용 특수 IFFT**에 가깝다.
- **`scalar` 인자**: `encode_internal`에서 넘기는 `fix = scale / sparse_slots` — 마지막 스테이지에서 샘플에 **실수 스케일**을 곱해 CKKS 스케일링과 일치시킨다.
- **`base_kernel`**: `sparse_slots`가 임계값 이하일 때 한 블록에서 공유 메모리로 여러 라운드를 처리; 크면 **`iter_kernel`** 이 여러 번 호출되어 큰 `n`을 분할 처리한다.

**이 구간에서의 의미**: 프로파일에서 **시간 비중 1위·2위**를 차지하는 것이 일반적이다. `ct_pt_matrix_mul_wo_pre`가 **인코딩을 곱셈 루프 안에서 반복**하기 때문이다.

---

## 3. `phantom::arith::decompose_array_uint64`

**파일**: `thirdparty/phantom-fhe/src/rns_base.cu`

**역할**

- IFFT 직후 GPU에 있는 **복소 계수 버퍼**(`cuDoubleComplex`)를 읽는다.
- 각 `(소모듈러스 인덱스, 스파스 계수 인덱스)`에 대해: 실수부/허수부를 **라운딩**하고, 부호에 따라 `q_i`에서의 **음수 표현**으로 바꾼 뒤 **Barrett reduction**으로 `\[0, q_i)` 범위의 `uint64` 한 개로 만든다.
- `sparse_ratio > 1`이면 같은 RNS 계수 뒤에 **0 패딩**을 넣어 전체 다항식 차수 `n`에 맞게 **슬롯 메시지를 “늘린” 다항식**으로 펼친다 (`encode_internal`의 `slots / sparse_slots` 비율과 연동).

**이 구간에서의 의미**: **부동소수 → 정수 RNS 계수**로 넘어가는 관문. 이후 연산은 전부 **정수 mod `q_i`** 다항식 연산이 된다.

---

## 4. `inplace_fnwt_radix8_phase1` / `inplace_fnwt_radix8_phase2`

**파일**: `thirdparty/phantom-fhe/src/ntt/fntt_2d.cu` (및 유사 래퍼에서 호출)

**역할**

- **Forward NTT** (Phantom 이름: **FNWT**): 계수 표현을 **NTT 평가 표현**으로 바꾼다. 각 소모듈러스 `q_i`마다 **negacyclic** 길이 `n` 변환을 **radix-8** 버터플라이로 쪼개, **phase1**(한 레이아웃/공유메모리 단계)과 **phase2**(다음 단계)로 나누어 실행한다.
- `encode_internal` 마지막의 `nwt_2d_radix8_forward_inplace` 등에서 **평문을 NTT 도메인**에 올릴 때 사용된다.

**이 구간에서의 의미**: **평문–암호문 곱**은 보통 NTT 도메인에서 **성분별 곱**으로 처리되므로, 인코딩 직후 이 커널들이 **곱셈 직전 준비**로 붙는다.

---

## 5. `multiply_rns_poly`

**파일**: `thirdparty/phantom-fhe/src/polymath.cu`

**역할**

- 동일 인덱스에서 `result[i] = (operand1[i] * operand2[i]) mod q_tower[i]` — 즉 **RNS 각 limb에서 계수별 곱** (Barrett/Shoup 등으로 감싼 mod mul).
- CKKS에서 **평문 × 암호문**은 다항식 곱에 해당하며, NTT 도메인에서는 **한 점에서의 곱**으로 단순화되어 이 커널 형태가 대량으로 등장한다.

**이 구간에서의 의미**: **실제 “행렬 곱”의 HE 쪽 곱셈**에 해당하는 GPU 작업의 상당 부분.

---

## 6. `add_rns_poly`

**파일**: `thirdparty/phantom-fhe/src/polymath.cu`

**역할**

- `result = (operand1 + operand2) mod q_i` per coefficient.
- 곱셈만 있는 것이 아니라, **여러 중간 다항식을 합치거나**, **relinearization / rescale의 일부**, 또는 **여러 열/블록 누적** 과정에서 덧셈이 섞이면 같이 잡힌다.

**이 구간에서의 의미**: `multiply_rns_poly`와 함께 **평가(evaluator)** 경로의 대역폭을 많이 쓰는 편.

---

## 7. `inplace_inwt_radix8_phase1` / `inplace_inwt_radix8_phase2`

**파일**: `thirdparty/phantom-fhe/src/ntt/intt_2d.cu`

**역할**

- **Inverse NTT** (INTT): NTT 도메인에서 **계수 도메인**(또는 다른 NTT 단계)으로 되돌리는 **radix-8** 2단계 커널.
- `divide_and_round_q_last_ntt` 등에서 **마지막 모듈러스 타워만 INTT**로 끌어내린 뒤 나눗셈을 하기 위해 `nwt_2d_radix8_backward_inplace` 다음 단계와 짝을 이루어 쓰인다 (`rns.cu` 주석: *Convert ci[last] to non-NTT form*).

**이 구간에서의 의미**: 본 TSV에서는 **`Instances`가 128로 매우 작고 비중 거의 0%** — 곱셈 루프 전체가 아니라 **특정 레벨에서 소수의 rescale / modulus 관리**만 NVTX 창 안에 살짝 걸린 것으로 해석할 수 있다.

---

## 8. `phantom::divide_and_round_reduce_q_last_kernel`

**파일**: `thirdparty/phantom-fhe/src/rns.cu` — `DRNSTool::divide_and_round_q_last_ntt` 흐름

**역할**

- **마지막 소모듈러스 `q_L`** 에 해당하는 암호문 계수를, 한 단계 줄인 베이스의 각 `q_j`에 대해 **`c_last mod q_j`** 형태로 **투영(reduce)** 한다.
- 즉 **차원 축소 전 준비**: “마지막 타워의 정보를 다른 소수들 위에서 어떻게 볼 것인가”를 정수 연산으로 처리.

---

## 9. `phantom::divide_and_round_ntt_inv_scalar_kernel`

**파일**: 동상 `rns.cu`, 같은 `divide_and_round_q_last_ntt` 시퀀스

**역할**

- INTT 및 reduce 이후, NTT 도메인에서 **`(c_j - (c_last mod q_j)) * q_L^{-1} mod q_j`** 에 해당하는 **성분별 스칼라 곱**(Shoup 곱)으로 마지막 모듈러스를 **반올림하며 제거**한다.
- SEAL/HE 표준 문헌의 **rescale / drop last modulus** 와 같은 계열 연산의 GPU 구현 조각이다.

**7–9번 함께**: 이 세 커널은 **한 번의 “마지막 모듈러스 제거” 파이프라인**에 연속으로 등장할 수 있으며, 본 프로파일에서는 호출 수가 적어 **누적 시간 비중은 미미**하다.

---

## 요약 파이프라인 (이 NVTX 구간 안에서)

1. **인코딩(웨이트 평문 준비)**  
   `bit_reverse_and_zero_padding` → `inplace_special_ifft_*` → `decompose_array_uint64` → `inplace_fnwt_radix8_phase*`

2. **평문–암호문 곱**  
   `multiply_rns_poly` (+ 필요 시 `add_rns_poly`)

3. **가끔의 레벨/모듈러스 처리**  
   `inplace_inwt_radix8_*` + `divide_and_round_*` (소수 인스턴스)

---

## 참고 소스 경로

| 커널 | 주요 정의 위치 |
|------|----------------|
| `bit_reverse_and_zero_padding` | `thirdparty/phantom-fhe/src/ckks.cu` |
| `inplace_special_ifft_*` | `thirdparty/phantom-fhe/src/fft.cu` |
| `decompose_array_uint64` | `thirdparty/phantom-fhe/src/rns_base.cu` |
| `inplace_fnwt_radix8_phase*` | `thirdparty/phantom-fhe/src/ntt/fntt_2d.cu` |
| `multiply_rns_poly`, `add_rns_poly` | `thirdparty/phantom-fhe/src/polymath.cu` |
| `inplace_inwt_radix8_phase*` | `thirdparty/phantom-fhe/src/ntt/intt_2d.cu` |
| `divide_and_round_*` | `thirdparty/phantom-fhe/src/rns.cu` |

---

*작성 기준: `ct_pt_micro_run1.kern_sum.mul_nvtx_clean.tsv` + Phantom-FHE 소스 정적 분석. 커널이 정확히 어떤 상위 API 한 줄에 대응하는지는 빌드 옵션·인라인·퓨전에 따라 달라질 수 있다.*
