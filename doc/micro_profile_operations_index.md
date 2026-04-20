# MOAI GPU 마이크로 프로파일 — 연산별 인덱스 (핸드오프용)

이 문서는 **지금까지 구현해 둔 Nsight 마이크로 벤치·NVTX·스크립트·산출물**을 한곳에 정리한 것이다. 새 채팅에서 **가중치(plain) 인코드 경로의 IFFT 비용을 줄이는 버전** 등을 이어갈 때, 이 파일을 프롬프트 컨텍스트로 붙이면 된다.

---

## 1. 빌드·도구 전제

| 항목 | 내용 |
|------|------|
| **NVTX** | `CMakeLists.txt`: `find_package(CUDAToolkit)` 후 `TARGET CUDA::nvtx3` 이면 `test`에 `CUDA::nvtx3` 링크 + `MOAI_HAVE_NVTX=1`. configure 로그에 `NVTX: linked CUDA::nvtx3` 확인. |
| **프로파일** | `nsys` (Nsight Systems), `python3`, 선택 **`matplotlib`** (`export_ct_pt_kern_sum.py` PNG). |
| **공통 플레이북** | [`micro_profile_playbook.md`](micro_profile_playbook.md) |
| **가속기 문서 절** | [`moai_accelerator_simulator_analysis.md`](moai_accelerator_simulator_analysis.md) §12 이후 (ct_pt, ct_pt_pre, ct_ct, softmax, gelu, layernorm, bootstrap, 플레이북). |
| **시뮬레이터 실행·env** | [`simulator_guide.md`](simulator_guide.md) — `MOAI_SIM_*`, `EngineModel`, primitive 마이크로 벤치. |

---

## 2. 진입점 (`MOAI_BENCH_MODE`)

모두 `src/test.cu` 의 `main()` 에서 `std::getenv("MOAI_BENCH_MODE")` 로 분기한다.

| `MOAI_BENCH_MODE` | 호출되는 테스트 | 비고 |
|-------------------|------------------|------|
| `boot` 또는 `bootstrap_micro` | `bootstrapping_test()` | 동일 바이너리 경로; 스크립트는 `bootstrap_micro` 사용 권장 |
| `ct_pt` | `ct_pt_matrix_mul_test()` | `ct_pt_matrix_mul_wo_pre` — 런타임에 `W`를 인코드하는 경로 |
| `ct_pt_pre` | `ct_pt_matrix_mul_w_preprocess_test()` | 루프로 `ecd_w` 사전 인코드 후 `ct_pt_matrix_mul` |
| `ct_ct` | `ct_ct_matrix_mul_test()` | col / diag 두 NVTX |
| `softmax_micro` | `softmax_micro_bench()` | `softmax_test` → `softmax_boot_test` 순서 |
| `softmax` | `softmax_test()` | 단독 |
| `softmax_boot` | `softmax_boot_test()` | 단독 |
| `gelu` | `gelu_test()` | |
| `layernorm` | `layernorm_test()` | |
| `sim_primitive` | `moai_sim_primitive_micro_bench(MOAI_SIM_PRIMITIVE)` | `MOAI_SIM_BACKEND=1` 필수. `MOAI_SIM_PRIMITIVE` 미설정 시 `all` |
| `sim_primitives` | 위와 동일 (`all`) | primitive마다 **별도 `.txt`** (`primitive_<tag>.txt` 등; `MOAI_SIM_REPORT_PATH` 해석은 `test_sim_primitives.cuh` 참고) |
| `sim_mul_plain` … `sim_modswitch` | 단일 primitive (`mul_plain`, `mul_ct`, `add_inplace`, `rescale`, `rotate`, `relin`, `modswitch`) | `test/sim/test_sim_primitives.cuh` |
| `sim_hybrid_ks_profile` | `moai_sim_hybrid_ks_profile_run()` — α 목록별 NTT/BConv 등 **구조적 카운트** + CSV | `source/sim/keyswitch_op_profile.h`; GPU 불필요 |

**Primitive sim 공통 env:** `MOAI_SIM_POLY_DEGREE`, `MOAI_SIM_NUM_LIMBS` (기본 **\|QP\|**=36 등, `sim_ckks_defaults.h`), **`MOAI_SIM_ALPHA`**, **`MOAI_SIM_NUM_LIMBS_COUNTS_QP`**(기본 1: \|Ql\|=\|QP\|−α), `MOAI_SIM_PRIMITIVE_LOOPS`, `MOAI_SIM_REPORT_PATH`(primitive는 **태그별 파일**; 단일 `moai_sim_report.txt`에 append하지 않음), `MOAI_SIM_REPORT_QUIET`, `MOAI_SIM_ENGINE_MODEL`. `rotate` / `relin` 은 coarse `SimTiming` 행 없이 **엔진 모델**만 증가.

---

## 3. NVTX 이름 (`nsys stats --filter-nvtx` 와 동일 문자열)

| 연산 맥락 | NVTX 문자열 | 정의 위치(요약) |
|-----------|-------------|------------------|
| Ct–Pt `wo_pre` | `moai:ct_pt_matrix_mul_wo_pre` | `test_ct_pt_matrix_mul.cuh` — `ct_pt_matrix_mul_wo_pre(...)` |
| Ct–Pt 사전 인코드 **W 루프** | `moai:ct_pt_pre_encode_w` | `test_ct_pt_matrix_mul.cuh` — `encoder.encode(W[i][j],…)` 전체 |
| Ct–Pt 사전 인코드 **곱만** | `moai:ct_pt_matrix_mul_pre_encoded` | 동 파일 — `ct_pt_matrix_mul(enc_ecd_x, ecd_w, …)` |
| Ct–Ct | `moai:ct_ct_matrix_mul_colpacking`, `moai:ct_ct_matrix_mul_diagpacking` | `test_ct_ct_matrix_mul.cuh` |
| Softmax | `moai:softmax_without_boot`, `moai:softmax_boot` | `test_softmax.cuh` |
| GeLU | `moai:gelu_v2_batch` | `test_gelu.cuh` |
| LayerNorm | `moai:layernorm` | `test_layernorm.cuh` |
| Bootstrap | `moai:bootstrap_prepare`, `moai:bootstrap_3` | `bootstrapping.cuh` |

---

## 4. 프로파일 스크립트 ↔ 산출 디렉터리

| 스크립트 (`src/scripts/`) | 기본 `MOAI_PROFILE_DIR` / `OUT_BASE` | `MOAI_BENCH_MODE` |
|---------------------------|--------------------------------------|-------------------|
| `profile_ct_pt_micro.sh` | `output/ct_pt/`, `ct_pt_micro` | `ct_pt` |
| `profile_ct_pt_pre_micro.sh` | `output/ct_pt_pre/`, `ct_pt_pre_micro` | `ct_pt_pre` |
| `profile_ct_ct_micro.sh` | `output/ct_ct/`, `ct_ct_micro` | `ct_ct` |
| `profile_softmax_micro.sh` | `output/softmax/`, `softmax_micro` | `softmax_micro` |
| `profile_gelu_micro.sh` | `output/gelu/`, `gelu_micro` | `gelu` |
| `profile_layernorm_micro.sh` | `output/layernorm/`, `layernorm_micro` | `layernorm` |
| `profile_bootstrap_micro.sh` | `output/bootstrap/`, `bootstrap_micro` | `bootstrap_micro` |

공통: `parse_nsys_cuda_kern_sum.py`, `export_ct_pt_kern_sum.py`, 인덱스 `*_index.tsv`.

---

## 5. Nsys·산출물에서 알아 둔 함정 (반드시)

1. **`nsys profile --force-overwrite=true -o "$stem"`**  
   없으면 기존 `*.nsys-rep`가 남고 새 캡처가 **`/tmp/nsys-report-*.nsys-rep`** 로만 가서, 스크립트가 읽는 경로와 달라 **NVTX 없음 / 잘못된 kern_sum** 이 된다.

2. **프로파일 직후 `rm -f "${stem}.sqlite"`** 후 `nsys stats --force-export=true`  
   오래된 SQLite가 NVTX 테이블 없이 재사용되는 문제를 막는다.

3. **`MOAI_NSYS_TRACE`** 에 **`nvtx`** 포함 (기본 `cuda,nvtx,osrt`). 빼면 필터 TSV가 비는 것과 유사한 증상.

---

## 6. Ct–Pt — `wo_pre`(forward) vs `ct_pt_pre`(벤치)

### 6.A Forward 최적화 타깃: **`ct_pt_matrix_mul_wo_pre`** (가중치 스칼라 인코드)

- **코드**: `src/include/source/matrix_mul/Ct_pt_matrix_mul.cuh` — `ct_pt_matrix_mul_wo_pre` 는 열 `i`마다 `W[0][i]…` 스칼라에 대해 **`encoder_local.encode(W[j][i], enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i, stream)`** 후 `multiply_plain`·`rescale` 을 반복한다.
- **왜 매번 슬롯 IFFT가 나가나**: `moai::Encoder::encode(double, …)` (`src/include/source/ckks_evaluator_parallel.cuh` 73–77행 부근)가 스칼라를 **`vector<double>(slot_count(), value)` 로 복제**한 뒤 `PhantomCKKSEncoder::encode` → `encode_internal` 을 탄다. 즉 **“모든 슬롯에 같은 실수”**임에도 **전 슬롯 길이 메시지로 취급**되어 `thirdparty/phantom-fhe/src/ckks.cu` 의 `encode_internal` 전체(비트리버스 패딩 → **`special_fft_backward`(슬롯 쪽 IFFT 계열)** → `decompose_array` → **`nwt_2d_radix8_forward_inplace`**)가 돈다.
- **의도하는 개선 방향(사용자 목표)**: 슬롯을 채우고 IFFT로 슬롯→다항식 변환하는 기존 경로 대신, **“모든 슬롯 상수”에 해당하는 평문을 RNS·NTT 도메인에 직접 쓰거나**, **한 번만 유도한 템플릿을 스케일/체인에 맞게 변형**하는 식으로 **`special_fft_backward` 비용을 제거·축소**한다. (구현은 Phantom `encode_internal` 쪽 fast path 추가 또는 MOAI `Encoder::encode(double,…)` 가 전 벡터를 만들지 않도록 바꾸는 등으로 갈 수 있음. 수학적으로는 CKKS 인코딩에서 상수 슬롯 메시지의 **계수/NTT 표현이 닫힌 형**이 있는지 SEAL/논문과 맞춰 검증 필요.)

### 6.B 마이크로 벤치: **`ct_pt_pre`** (사전 인코드 루프 vs 곱만 NVTX)

- **테스트**: `test_ct_pt_matrix_mul.cuh` — `ct_pt_matrix_mul_w_preprocess_test()` (루프에서 `ecd_w` 생성 후 `ct_pt_matrix_mul`).
- **곱**: `Ct_pt_matrix_mul.cuh` — `ct_pt_matrix_mul`.
- **`rescale` 시 마지막 limb만 잠깐 INTT**: `rns.cu` 의 `divide_and_round_q_last_ntt` — **슬롯 IFFT와 무관.**

#### 프로파일에서 보이는 것

- 전체 `kern_sum`: X 배치 인코드 + **`moai:ct_pt_pre_encode_w`**(6.A와 동일한 `encode` 파이프라인이 **루프로 많이** 돈 것) + **`moai:ct_pt_matrix_mul_pre_encoded`**.
- NVTX TSV: `*.kern_sum.nvtx_encode_w.tsv`, `*.kern_sum.pre_enc_nvtx.tsv`.

#### 메모리

- **`MOAI_CT_PT_PRE_MICRO`**: 스크립트 기본 `1`이면 문제 크기 축소.

---

## 7. 기타 커널 분석 MD (예시)

`output/` 아래 연산별로 `*_kernel_reference.md`, `*_kern_sum_nvtx_*_kernel_analysis.md` 등이 있다 (ct_ct, softmax, gelu, layernorm, bootstrap, ct_pt 등). 전체 목록은 `output/<op>/` 디렉터리를 보면 된다.

---

## 8. 새 채팅에 넣을 때 추천 한 줄 프롬프트 예시

> `MOAI_GPU/doc/micro_profile_operations_index.md` §6.A 기준으로, **`ct_pt_matrix_mul_wo_pre`** 가 스칼라 `W[j][i]` 마다 부르는 `Encoder::encode(double,…)` 가 **전 슬롯 벡터 + `ckks.cu` `encode_internal` 의 `special_fft_backward`** 를 타지 않도록, **상수 슬롯 메시지를 다항식(RNS/NTT)에 직접 올리는 fast path** 를 설계·구현하고, `profile_ct_pt_micro.sh` 의 **`moai:ct_pt_matrix_mul_wo_pre`** NVTX 전후로 `kern_sum`·IFFT 커널 비중이 줄었는지 검증해 달라. (필요 시 `ct_pt_pre` 의 `nvtx_encode_w` 벤치는 동일 encode 파이프라인 비교용으로 유지.)

---

*저장 위치: 저장소 `MOAI_GPU/doc/`. Cursor 규칙: `.cursor/rules/moai-micro-profile.mdc` + `doc/micro_profile_playbook.md`.*
