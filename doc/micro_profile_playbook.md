# MOAI GPU 마이크로 프로파일 공통 플레이북

새 연산(GeLU, LayerNorm, Bootstrapping, …)을 **ct_pt / ct_ct / softmax / bootstrap_micro** 와 동일한 방식으로 다룰 때의 체크리스트이다.

## 1. 벤치 진입점 (`src/test.cu`)

- `MOAI_BENCH_MODE=<이름>` 분기 추가.
- 한 프로세스에 **여러 NVTX 슬라이스**가 필요하면 `*_micro_bench()` 에서 테스트를 순서대로 호출 (예: `softmax_micro_bench`).
- **한 구간만**이면 단일 테스트 함수만 호출해도 된다 (예: `gelu_test`).

## 2. NVTX (`MOAI_HAVE_NVTX`)

- 테스트 헤더에 `#include <nvtx3/nvToolsExt.h>` (가드: `#if defined(MOAI_HAVE_NVTX)`).
- 측정할 **핵심 GPU 구간**만 `nvtxRangePushA("moai:<고유이름>")` … `Pop()` 으로 감싼다.
- OpenMP 병렬 구간이면 **병렬 블록 직전/직후**에 `cudaDeviceSynchronize()` 를 두어 NVTX와 GPU 타임라인이 맞도록 한다.
- 이름 규칙: **`moai:` 접두사** + 짧은 스네이크 케이스 (Nsight `filter-nvtx` 와 동일 문자열).

## 3. 프로파일 셸 스크립트 (`src/scripts/profile_<이름>_micro.sh`)

- `profile_softmax_micro.sh` 또는 `profile_ct_ct_micro.sh` 를 복사해 수정한다.
- 고정 요소:
  - `REPO_ROOT`, `BUILD_DIR`, `TEST_EXE`, `OUT_DIR` (기본 `output/<연산>/`), `OUT_BASE` (`<연산>_micro`).
  - `nsys profile --force-overwrite=true -o ...` — 기존 `*.nsys-rep` 가 있으면 nsys가 **조용히 `/tmp/nsys-report-*.nsys-rep` 로만** 새 캡처를 쓰고, 스크립트가 가리키는 경로는 **옛 파일**이라 NVTX/`kern_sum` 이 어긋날 수 있다.
  - `nsys profile` 시 `export MOAI_BENCH_MODE=...`.
  - `nsys stats ... cuda_gpu_kern_sum` → 전체 `*.kern_sum.tsv`.
  - 각 NVTX마다 `--filter-nvtx "moai:..."` → `*.kern_sum.nvtx_<suffix>.tsv` (파일명만 팀에서 통일).
  - `python3 parse_nsys_cuda_kern_sum.py` 로 인덱스용 요약 문자열.
  - `MOAI_KERN_EXPORT=1` 이면 `export_ct_pt_kern_sum.py --out-dir "$OUT_DIR"` 에 **전체 + 각 nvtx tsv** 를 넘겨 `*_clean.tsv` 및 bar/pie PNG 생성.
- 환경 변수: `MOAI_PROFILE_DIR`, `MOAI_PROFILE_OUT`, `MOAI_PROFILE_REPEAT`, `MOAI_NSYS_TRACE`, `MOAI_NSYS_STATS_FILTER_NVTX`, `MOAI_KERN_EXPORT` 등 기존 스크립트와 동일 패턴 유지.

## 4. 그래프

- `matplotlib` 설치된 환경에서 프로파일 스크립트가 끝나면 PNG가 `OUT_DIR` 에 생긴다.
- 수동:  
  `python3 src/scripts/export_ct_pt_kern_sum.py --out-dir output/<연산> output/<연산>/*.kern_sum.nvtx_*.tsv`

## 5. 커널 분석 Markdown

- 산출: `output/<연산>/<stem>_kern_sum_nvtx_<suffix>_kernel_analysis.md`.
- 입력: 대응하는 `*_clean.tsv` (표준 헤더 + `ShortName` 열).
- 내용: NVTX 맥락 한 단락, **ShortName 요약 표**, Phantom/MOAI 소스 경로가 알려진 커널별 절, 파이프라인 요약.
- Ct–Ct·softmax 문서를 **재사용·참조**해 동일 Phantom 커널 설명을 복붙하지 않도록 cross-link 해도 된다.

## 6. 문서 인덱스

- `doc/moai_accelerator_simulator_analysis.md` 에 **짧은 절**로 스크립트 경로·`output/` 위치·이 플레이북 링크를 추가한다.

---

관련 스크립트 예: `profile_ct_pt_micro.sh`, `profile_ct_pt_pre_micro.sh`, `profile_ct_ct_micro.sh`, `profile_softmax_micro.sh`, `profile_gelu_micro.sh`, `profile_layernorm_micro.sh`, `profile_bootstrap_micro.sh`.
