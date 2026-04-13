# MOAI 가속기 시뮬레이터 설계 — Neo 방식 참조 가능성 분석

본 문서는 [moai_accelerator_simulator_analysis.md](moai_accelerator_simulator_analysis.md)의 MOAI_GPU 구조 분석을 전제로, Neo 저장소의 시뮬레이터 패턴([Neo/doc/simulator_architecture.md](../../Neo/doc/simulator_architecture.md))을 **MOAI CKKS 가속기 시뮬**에 이식·참조할 수 있는지 정리한다.

---

## 1. 결론 요약

| 질문 | 답 |
|------|-----|
| Neo 코드를 **그대로 포크**해 MOAI에 쓸 수 있는가? | **거의 없음.** 도메인(3DGS 타일·정렬·래스터)과 MOAI(FHE 레이어·모듈러스 체인·부트스트랩)가 다르다. |
| Neo **방식**이 설계에 도움이 되는가? | **된다.** “이종 클럭 틱 + Ramulator + 연속/이산 메모리 API + 위상기 FSM + 오프라인 트레이스 + YAML 시나리오 + 변형 바이너리(풀/축소)”는 MOAI에 그대로 매핑 가능하다. |
| 가장 가치 있는 빌록 | **DRAM/HBM 타이밍과 코어 사이클의 분리**, **요청 완료를 기다리는 FSM**, **블랙박스 고비용 연산**(Neo의 정렬·서브타일 ≈ MOAI의 `bootstrap_3`·inverse 루프)의 **계층적 모델**. |

---

## 2. Neo 패턴과 MOAI 분석 문서의 1:1 대응

### 2.1 시간축: 이종 클럭 (`simulator_t`)

- **Neo**: DRAM 클럭 vs CORE 클럭을 LCM/GCD로 맞춰 `dram_wrapper->tick()` / `core->tick()`을 번갈 수행.
- **MOAI 활용**:  
  - **HBM/DRAM** (키·rotation material·대형 ct 버퍼 스트리밍) vs **가속기 PE/NTT 파이프** 클럭을 분리한다.  
  - FHE 가속기에서 메모리가 자주 병목이므로, MOAI 분석 §4.2의 `ciphertext_device_bytes`·live ct 수와 결합하면 **대역폭 바운드 구간**을 Neo와 같은 방식으로 드러내기 쉽다.

### 2.2 메모리: Ramulator + 래퍼 + (선택) 캐시

- **Neo**: `dram_t` + `dram_wrapper_t`의 `continuous_dram_access` / `discrete_dram_access`, 선택적 `cache_t`.
- **MOAI 활용**:  
  - **연속 스트림**: RNS 다항식 계수·큰 plain 배열의 순차 DMA.  
  - **이산 접근**: 768개 ct처럼 **논리적으로 흩어진 버퍼**를 캐시라인 단위로 모을 때 Neo의 `discrete_dram_access` 패턴이 유용하다.  
  - Ramulator는 **LPDDR4뿐 아니라 HBM** 백엔드도 선택 가능(Neo `dram.cc`의 `HBM` 팩토리 등) — MOAI 분석이 말하는 대형 메모리·키 저장과 정합.

### 2.3 입력: 오프라인 트레이스 (`poc.trace` 역할)

- **Neo**: 실제 3DGS를 돌리지 않고 `poc.trace`로 타일·가우시안·서브타일 통계를 주입.
- **MOAI에 대응하는 “트레이스” 최소 필드** (분석 문서 §5–§8 기반):

  - 레이어 스텝 ID, 헤드 인덱스(0–11 또는 병렬 시나리오 플래그).
  - 연산 **종류**: `mod_switch`, `multiply_plain`, `ct_ct_matrix_mul_*`, `bootstrap_3`, `square`/`relinearize`/`rescale` 카운트, inverse `iter`, LN/GELU 서브루틴 태그.
  - **체인/depth**: `pre_att_drops`, `boot_level`, `MOAI_LN_BOOTSTRAP_VARIANT` 등으로 결정된 시작·끝 depth (분석 §3.2, §7).
  - **바이트**: ct/pt 하나당 크기 × 동시 live 수(분석 §10 항목 4).
  - (선택) NTT/키스위치 횟수 — 분석 §5 후반: Phantom 훅 또는 블랙박스 latency.

트레이스는 **GPU에서 한 번 프로파일**하거나, **MOAI 환경 변수 시나리오**(§7)를 고정한 채 **정적 스케줄러가 DAG를 펼쳐** 생성해도 된다. Neo의 `generate_simulator_yaml.py`와 같이 **YAML로 시나리오를 재생**하는 형태가 MOAI 분석 §7의 “recipe + env 오버라이드”와 잘 맞는다.

### 2.3.1 GPU에서 뽑는 트레이스 — Nsight Systems · CUPTI 기준 최소 필드

MOAI는 이미 HBM GPU에서 돌아가므로, **실측 타임라인**을 Neo의 `poc.trace`처럼 **오프라인 중간 표현**으로 줄일 때 아래를 최소 집합으로 두면 Ramulator 쪽(`continuous` / `discrete` 스타일)과 맞추기 쉽다.

**공통 메타 (모든 이벤트 행)**

| 필드 | 용도 |
|------|------|
| `t_start`, `t_end` (또는 `duration_ns`) | 코어 FSM과의 **순서·병렬 구간**; 동일 stream 내 전역 순서 |
| `stream_id` | Phantom `stream_pool`·다중 stream과 대응; **자원 경쟁** 모델 시 필수 |
| `device_id` | 멀티 GPU 대비(단일면 생략 가능) |
| `correlation_id` / `seq` | Nsight·CUPTI가 주는 상관 ID로 **memcpy ↔ kernel** 묶기 |

**커널 (`cudaLaunchKernel` 등)**

| 필드 | 용도 |
|------|------|
| `kernel_name` (가능하면 demangle) | MOAI 논리 연산과 매핑; 이름이 불투명하면 **호스트 훅**으로 `logical_op` 태그 추가 |
| `grid_x/y/z`, `block_x/y/z` | 워크 단위 스케일; 나중에 **사이클 애널리틱**에 곱하기 |
| `registers`, `static_smem` (선택) | 점유·파티션 감이 필요할 때 |

**메모리 (Ramulator 입력으로 직결되는 층)**

| 필드 | 용도 |
|------|------|
| `memcpy`: `src_kind`, `dst_kind` (`H2D`/`D2D`/`D2H`) | HBM 트래픽 방향; Neo식 연속 스트림의 상한 추정 |
| `memcpy`: `byte_count`, `alignment` (선택) | `continuous_dram_access` 길이·캐시라인 정렬 모사 |
| `kernel` 측 **바이트** | 이상적: CUPTI **메트릭**(`dram__bytes_read.sum` 등) 또는 서브패스; 없으면 MOAI 측 **버퍼 크기 + R/W 횟수**로 **상한 추정** 후 `discrete_dram_access` 리스트 생성 |
| `dev_ptr` / `allocation_id` (선택) | 동일 버퍼 재사용·live range; 분석 문서 §10의 **동시 live ct**와 연결 |

**동기화**

| 필드 | 용도 |
|------|------|
| `cudaStreamSynchronize` / `cudaDeviceSynchronize` / `Event` | 파이프라인 **배리어**; 시뮬에서 “한 위상 종료” 트리거로 쓰기 좋음 |

**도구별 메모**

- **Nsight Systems**: 타임라인 CSV/보내기에서 위 필드 대부분 확보 가능. 커널별 HBM 바이트는 **반드시** 나오지 않을 수 있어, 1차는 **memcpy + MOAI가 이미 갖는 ct 바이트 합**으로 보강하는 현실적 타협이 많다.
- **CUPTI (Activity API / PC Sampling)**: 커널·memcpy 이벤트에 바이트·상관 ID를 붙이기 유리. **오버헤드**가 크므로 “한 레이어 짧은 런” + 시나리오 고정을 권장.

**Neo `poc.trace`와의 정렬**

- GPU 원시 로그는 **주소·순서가 실칩과 1:1이 아닐** 수 있음(L2, 병합 접근 등). 시뮬 목적이 **상대 비교·설계 스윕**이면 위 최소 필드 + **논리 DAG**(§2.3 본문)가면 충분하고, “절대 사이클 일치”는 **호스트 계측 + 합성 트레이스**를 병행하는 편이 안전하다.

### 2.4 코어: 위상기 FSM + 병렬 풀 (`core_t` / `*_phase_t`)

- **Neo**: `common_phase` / `reuse_phase` / `render_phase`가 동시에 `tick`되고, DRAM 완료 ID(`is_finished(m_id)`)로 상태 전이.
- **MOAI 매핑 예시** (하나의 설계 스케치):

  | Neo 위상 | MOAI에 대응 가능한 추상 위상 (예시) |
  |----------|-------------------------------------|
  | common (전처리·전역 정렬) | **인코딩·키스위치·어텐션 QKV·체인 드롭** (`batch_input`, `mod_switch` 연쇄) |
  | reuse (적응 정렬) | **QK / softmax / V 결합** 등 depth 비대칭 경로 (`single_att_block`, V depth cap) |
  | render | **FFN·GELU·후반 선형·(LN2)** — 다항·BSGS를 **고정 사이클 서브모델**로 두는 단계 |

실제 하드웨어가 **파이프라인 단계가 다르면** Neo처럼 **별도 `*_phase_t`로 쪼개고**, 각 위상이 같은 `dram_wrapper`를 공유하며 **경쟁하는 구조**로 두면 된다.

### 2.5 고비용 연산의 2단계 모델 (Neo의 approximation / precise)

- **Neo**: 정렬을 “근사 다패스 DRAM” + “청크 단위 precise + 사이클 카운트”로 나눈다.
- **MOAI**: 분석 §10 항목 3과 동일하게  
  1. **`bootstrap_3`를 블랙박스 고정 사이클** (입출력 depth만 맞춤)  
  2. 내부를 펼친 **미세 모델** (Bootstrapper 내부 NTT·modup 등)  
으로 층을 나누면, Neo의 **GLOBAL_SORTER_STATE_APPROXIMATION vs PRECISE**와 같은 **정밀도·개발 단계 트레이드오프**를 재현할 수 있다.

### 2.6 변형 바이너리 (`neo` / `neo_only_s` / `neo_s`)

- **Neo**: 풀 파이프라인 vs 정렬만 vs 정렬+후처리(postprocessor).
- **MOAI 매핑** (분석 §3.1 `MOAI_BENCH_MODE`와 자연스럽게 대응):

  | Neo 타깃 | MOAI 시뮬 변형 아이디어 |
  |----------|-------------------------|
  | `neo_only_s` | **부트스트랩 단독** 또는 **ct–ct 단독** (`boot`, `ct_ct`) 만 DRAM+연산 모델 |
  | `neo_s` | **부트스트랩 후 modulus/메타데이터 갱신**을 별도 위상으로 두는 설계(소프트웨어 후처리 vs 통합 가속) 비교 |
  | `neo` (full) | **`single_layer_test` 전체 DAG** + 환경 변수 시나리오 |

---

## 3. Neo 방식으로 **커버하기 어려운** MOAI 측면

다음은 Neo 시뮬의 강점 밖이므로 **별도 모듈**로 두는 것이 좋다.

1. **CKKS 노이즈·정확도** — 분석 §11: 사이클-only 시뮬과 분리된 오차/스케일 모델.
2. **Phantom CUDA 스케줄** — Neo는 CPU 측 C++ 이벤트 루프; MOAI는 `stream_pool`·OpenMP(분석 §9)로 **실제 GPU와의 gap**이 있다. 시뮬은 “이상적 병렬 vs 순차 헤드”를 **명시적 시나리오**로 두어야 한다.
3. **모듈러스 체인의 전역 의존성** — Neo의 타일 큐보다 **레이어 전역 depth 예산**이 강하므로, FSM 상태에 **chain_index / noise budget**을 넣는 편이 Neo 원본보다 중요하다.

---

## 4. 권장 활용 순서 (실무)

1. **MOAI 분석 문서 §5 DAG + §7 env**를 “논리 트레이스 스키마”로 고정한다.  
2. **Neo식**: YAML(`DRAM`/`CORE`/`OTHER`) + `elapsed_unit_time` 루프 + `dram_wrapper` 유사 API를 **새 C++ 또는 Python**으로 최소 구현한다 (Neo 서브트리 복사는 선택).  
3. **Ramulator**: LPDDR4로 시작해 필요 시 HBM cfg로 스케일링 실험.  
4. **검증**: 분석 §5·§10이 가리키는 `results.csv` / `single_layer_results.csv`와 **총 사이클·총 바이트**를 맞춘다.  
5. **Neo `neo_hs`처럼**: 동일 DAG에 **부트스트랩 placement variant**만 바꾼 두 번째 바이너리로 **설계 공간 스윕**을 자동화한다.

---

## 5. 관련 문서

- MOAI: [moai_accelerator_simulator_analysis.md](moai_accelerator_simulator_analysis.md), [layernorm_moai_summary.md](layernorm_moai_summary.md), [gelu_hmult_bsgs.md](gelu_hmult_bsgs.md)
- Neo 시뮬 요약: [Neo/doc/simulator_architecture.md](../../Neo/doc/simulator_architecture.md)

---

*요지: Neo는 MOAI의 “연산 내용” 템플릿이 아니라, **메모리 정확 시뮌 + 이벤트 기반 코어 + 트레이스 주입 + 단계별 모델 축소**라는 **시뮬레이터 엔지니어링 패턴**의 참고 구현으로 쓰는 것이 가장 효율적이다.*
