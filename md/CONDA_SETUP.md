# MOAI GPU Conda Environment Setup Guide

이 문서는 `MOAI_GPU` 프로젝트를 conda 환경에서 빌드하고 실행하기 위한 최소 환경 셋업 절차를 정리한 문서다.  
기본 요구사항은 `thirdparty/phantom-fhe/README.md`, `MOAI_GPU/README.md`, 그리고 현재 프로젝트의 `CMakeLists.txt`를 기준으로 정리했다.
특히 이 버전은 사용 GPU를 **NVIDIA RTX PRO 6000 Blackwell Workstation Edition**으로 가정하고 정리했다.

## 1. 권장 환경

- OS: Ubuntu/Linux
- GPU: `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- CUDA: `12.8+` 권장
- CMake: `>= 3.20`
- GCC / G++: `11.x`
- Python: conda 환경용 `3.10` 권장

현재 저장소 기준으로 다음 사항이 중요하다.

- 루트 `CMakeLists.txt`는 CUDA 컴파일러를 명시적으로 주지 않으면 `/usr/local/cuda-12.8/bin/nvcc`를 우선 사용하고, 없으면 `/usr/local/cuda-11.8/bin/nvcc`로 fallback 하도록 수정했다.
- 루트 `CMakeLists.txt`의 기본 `CMAKE_CUDA_ARCHITECTURES`는 이제 `120`이다.
- 프로젝트 README 기준 기존 테스트 환경은 `NVIDIA H200`, `A100`이었다.
- 링크 라이브러리로 `ntl`, `gmp`, `OpenMP`가 필요하다.

즉, **현재 저장소 기본값도 RTX PRO 6000 Blackwell 기준에 맞춰 반영된 상태이며, 필요하면 명령행 `-D...` 옵션으로 덮어쓸 수 있다.**

## 2. Blackwell 기준 핵심 메모

`RTX PRO 6000 Blackwell`을 기준으로 보면, 현재 저장소 설정에서 특히 아래 두 부분을 기억하면 된다.

1. 기본 CUDA 컴파일러는 `12.8` 우선, `11.8` fallback 구조다.
2. 기본 `CMAKE_CUDA_ARCHITECTURES`는 `120`이다.

정리하면 다음 전략이 가장 안전하다.

- **권장 경로:** `CUDA 12.8+` 환경에서 Blackwell 대응 `nvcc`를 사용
- **문서 기준 권장 아키텍처:** 가능하면 `CMAKE_CUDA_ARCHITECTURES=120` 우선 검토
- **보수적 대안:** 정확한 native cubin 지원이 애매하면 PTX 포함 빌드가 되도록 유지하고 호환성 테스트 수행

NVIDIA Blackwell 호환성 가이드 기준으로, CUDA 11.8-12.7로 빌드한 애플리케이션도 **PTX가 포함되어 있으면** Blackwell에서 JIT 방식으로 동작할 수 있다. 다만 **Blackwell native 지원과 튜닝까지 고려하면 CUDA 12.8+ 쪽이 더 적합하다.**

## 3. 사전 확인

아래 항목이 먼저 준비되어 있어야 한다.

### CUDA 설치 확인

```bash
which nvcc
nvcc --version
```

현재 저장소는 명시적 override가 없을 때 `/usr/local/cuda-12.8/bin/nvcc`를 우선 사용하도록 수정되어 있다.  
`RTX PRO 6000 Blackwell`을 기준으로는 `CUDA 12.8+` 사용을 권장한다.

이 경우 아래 둘 중 하나가 필요하다.

1. **저장소 기본 설정 사용**: CUDA 12.8 경로가 있으면 그 값을 그대로 사용
2. **명시적 override 사용**: `-DCMAKE_CUDA_COMPILER=...`로 실제 CUDA 경로를 직접 지정

### GPU 아키텍처 확인

```bash
nvidia-smi
```

현재 저장소 기본 설정은 `120`이다.

- A100: `80`
- H100/H200 계열: `90`
- RTX PRO 6000 Blackwell: `120` 기준으로 문서를 정리했다.

권장 순서는 다음과 같다.

1. `nvidia-smi`로 GPU 모델 확인
2. `nvcc --version`으로 현재 툴체인 확인
3. Blackwell 대응 `nvcc`라면 `CMAKE_CUDA_ARCHITECTURES=120` 우선 검토
4. 확실하지 않으면 `PHANTOM_USE_CUDA_PTX=ON` 상태를 유지하고 JIT 호환성까지 확인

## 4. conda 환경 생성

아래 예시는 `moai`라는 이름의 새 conda 환경을 만드는 절차다.

```bash
conda create -n moai python=3.10 -y
conda activate moai
```

## 5. conda 패키지 설치

빌드에 필요한 기본 패키지를 설치한다.

```bash
conda install -c conda-forge \
  cmake \
  ninja \
  gcc_linux-64=11 \
  gxx_linux-64=11 \
  gmp \
  ntl \
  make \
  pkg-config \
  -y
```

설치 후 컴파일러가 conda 환경의 GCC/G++를 우선 사용하도록 다음 값을 확인한다.

주의: conda-forge의 `gcc_linux-64`, `gxx_linux-64`는 plain `gcc`, `g++`를 덮어쓰지 않는 경우가 많다.  
즉, `gcc --version`, `g++ --version`만 보면 여전히 시스템 기본 컴파일러(예: Ubuntu 13.x)가 보일 수 있다.

```bash
echo "$CC"
echo "$CXX"
$CC --version
$CXX --version
```

여기서 `x86_64-conda-linux-gnu-gcc`, `x86_64-conda-linux-gnu-g++` 경로가 잡히고 버전이 `11.x`로 보이면 된다.

## 6. 환경 변수 설정

conda 환경에서 라이브러리 탐색 경로가 꼬이지 않도록 아래 값을 지정하는 것을 권장한다.

```bash
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX:$CMAKE_PREFIX_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

한 세션에서 바로 적용하려면 위 명령을 실행하고, 자주 사용할 경우 `~/.bashrc` 또는 conda activation script로 옮기면 된다.

Blackwell용 CUDA를 따로 설치했다면 아래도 함께 맞춰준다.

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

## 7. 저장소 준비

`phantom-fhe`는 서브디렉터리로 포함되어 있으므로 `MOAI_GPU` 루트에서 작업한다.

```bash
cd /home/jyg/projects/MOAI_GPU
```

## 8. 빌드

프로젝트 README 기준 기본 빌드 절차는 아래와 같다.

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### RTX PRO 6000 Blackwell 기준 권장 빌드

현재 저장소의 기본 `CMakeLists.txt`는 `CUDA 12.8` 우선, `CMAKE_CUDA_ARCHITECTURES=120` 기본값으로 동작한다.  
`RTX PRO 6000 Blackwell`을 기준으로는 아래처럼 **툴체인 경로와 아키텍처를 명시**하는 쪽이 더 명확하다.

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120

cmake --build build -j
```

### 현재 저장소 설정을 그대로 둘 때

기본 `CMakeLists.txt`는 이미 Blackwell 기준값을 기본으로 사용한다.  
다만 아래 둘 중 하나로 환경을 더 명시적으로 고정할 수 있다.

1. `CMakeLists.txt`를 직접 수정
2. 필요 시 해당 값을 프로젝트 정책에 맞게 정리한 뒤 다시 configure

예시:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120
```

현재는 명령행 `-DCMAKE_CUDA_ARCHITECTURES=...`를 주면 그 값이 기본값보다 우선한다.

또한 `PHANTOM_USE_CUDA_PTX`가 `ON`이면 PTX 포함 빌드 쪽에 유리하므로, Blackwell 초기 호환성 확인 단계에서는 유지하는 편이 안전하다.

## 9. 실행

README 기준 실행 예시는 다음과 같다.

```bash
cd build
OMP_NUM_THREADS=4 ./test
```

추가 메모:

- `OpenMP`는 필수다.
- README에는 `OMP_NUM_THREADS=4` 예시가 있지만, 실제 최적값은 CPU/GPU 조합에 따라 `4~8` 사이에서 조정하는 것이 좋다.

## 10. 빠른 점검 체크리스트

아래 항목이 모두 맞으면 대부분 빌드 준비가 끝난 상태다.

- `conda activate moai` 상태다.
- `nvcc --version`이 의도한 CUDA 버전을 가리킨다.
- `echo $CC`, `echo $CXX`가 conda 컴파일러 경로를 가리킨다.
- `$CC --version`, `$CXX --version`이 `11.x`다.
- conda 환경에 `cmake`, `gmp`, `ntl`가 설치되어 있다.
- GPU 아키텍처가 현재 CMake 설정과 맞는다.
- `RTX PRO 6000 Blackwell` 기준이면 기본 아키텍처가 `120`으로 잡힌다.

## 11. 추천 전체 명령 예시

처음부터 한 번에 진행하려면 아래 순서로 실행하면 된다.

```bash
conda create -n moai python=3.10 -y
conda activate moai

conda install -c conda-forge \
  cmake \
  ninja \
  gcc_linux-64=11 \
  gxx_linux-64=11 \
  gmp \
  ntl \
  make \
  pkg-config \
  -y

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX:$CMAKE_PREFIX_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

cd /home/jyg/projects/MOAI_GPU
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j

cd build
OMP_NUM_THREADS=4 ./test
```

## 12. 문제 발생 시 먼저 볼 포인트

### `nvcc`를 찾지 못하는 경우

- `which nvcc`와 `nvcc --version` 먼저 확인
- Blackwell 기준이면 `/usr/local/cuda-12.8/bin/nvcc` 같은 실제 경로 확인
- 저장소 기본 탐색 경로에 원하는 CUDA 버전이 없으면 `-DCMAKE_CUDA_COMPILER=...`로 직접 지정

### `ntl` 또는 `gmp` 링크 오류가 나는 경우

- conda 환경에 `ntl`, `gmp` 설치 여부 확인
- `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` 적용 여부 확인

### GCC 버전 충돌이 나는 경우

- `gcc --version`, `g++ --version`만 보지 말고 `echo $CC`, `echo $CXX` 먼저 확인
- `$CC --version`, `$CXX --version`이 `11.x`인지 확인
- 시스템 기본 GCC가 먼저 보이더라도 `CC`, `CXX`가 conda compiler wrapper를 가리키면 빌드에는 문제없을 수 있다.

### GPU 아키텍처 관련 컴파일 문제가 나는 경우

- 현재 GPU 모델 확인
- `CMAKE_CUDA_ARCHITECTURES` 값을 GPU에 맞게 조정
- `RTX PRO 6000 Blackwell` 기준이면 기본값 `120`이 잡히는지 확인
- 애매하면 `PHANTOM_USE_CUDA_PTX=ON` 유지 후 `CUDA_FORCE_PTX_JIT=1`로 호환성 점검

## 13. 참고

- Phantom FHE README: `thirdparty/phantom-fhe/README.md`
- MOAI GPU README: `README.md`
- 추가 Phantom 문서: [Phantom FHE GitBook](https://encryptorion-lab.gitbook.io/phantom-fhe/)
- NVIDIA RTX PRO 6000 Blackwell Workstation Edition: [NVIDIA 제품 페이지](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000.md)
- NVIDIA Blackwell Compatibility Guide: [CUDA Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)
