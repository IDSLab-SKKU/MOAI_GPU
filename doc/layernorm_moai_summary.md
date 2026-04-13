# MOAI GPU LayerNorm (`layernorm.cuh`) — 정리 및 depth 줄이기

대상: `MOAI_GPU/src/include/source/non_linear_func/layernorm.cuh` 의 `layernorm` / `invert_sqrt` 경로.  
(대화에서 다룬 내용을 코드 기준으로 요약한다.)

---

## 1. 구현이 하는 일 (수식 스케치)

- **합**: `ave_x = sum_i x[i]` (768개 암호문 각각이 한 슬롯/패킹에 대응한다고 가정한 구조).
- **스케일**: `nx[i] = x[i] * n` (`n=768`, 슬롯 마스크 `bias_vec`로 활성 슬롯에만 `ecd_n[i]=768`).
- **분산에 해당하는 양 `var`**:  
  각 `(nx - ave_x)`를 제곱해 모두 더한 뒤, **중간에 항마다 `relinearize/rescale` 하지 않고** 부분합(48×16) 후 한 번만 `relinearize` + `rescale`, 이어서 `1/(n^2)` 평문 곱.  
  순수하게는 \(\sum_i (x_i-\mu)^2\) 에 가깝고, 이후 `gamma/sqrt(n)` 등과 맞물려 **표준 LayerNorm의 \(\frac{x-\mu}{\sqrt{\mathrm{Var}}}\)** 와 같은 스케일의 정규화가 되도록 맞춘 형태(직접 \(\sigma^2/n\) 한 번에 나누는 것과는 표기가 다름).
- **정규화**: `invert_sqrt(var)` 로 역제곱근에 해당하는 값을 구한 뒤, `(nx - ave_x)`에 곱하고 `gamma/sqrt(n)`, `beta` 적용.

---

## 2. Lazy relinearization / rescale (분산 블록)

- 제곱 루프 안에서는 `relinearize` / `rescale` 가 **주석 처리**되어 있음.
- **전체 제곱편차 합 `var`를 만든 뒤** 한 번 `relinearize` + `rescale`.
- 목적: **곱 depth·연산 횟수 절감** (matmul의 “sum 끝에 relin/rescale”과 같은 철학).

---

## 3. `invert_sqrt` (Newton + Goldschmidt)

- **`initGuess`**: `evalLine`으로 **선형 근사** \(y_0 \approx a x + b\) — **`multiply_plain`만** (ct–ct 0).
- **`newtonIter` (기본 4회)**: 역제곱근용 뉴턴  
  \(y \leftarrow \frac{3}{2}y - \frac{1}{2} x y^3\)  
  (회당 `square` 1 + `multiply` 2 등 ct–ct 위주).
- **`goldSchmidtIter` (기본 2회)**: 표준 Goldschmidt 루프; **논문에서 말하는 \(E^{2^k}\) 전개·product-tree 식 “언롤링 병렬” 구현은 아님**.
- 호출: `invert_sqrt(var, 4, 2, ...)`.

파일 주석에 **“depth need = 20 (current version)”** 이 있으며, **ct–ct만 잡는 다항식 경로**로는 대략 **분산 제곱 1 + `invert_sqrt` 내부 ~19** 정도로 읽는 해석이 가능(출력 슬롯마다 `inv_sqrt`와의 ct–ct 곱 1은 별도).

---

## 4. \(\sigma^2 + \epsilon\) 의 \(\epsilon\)

- **현재 코드에는 `var`에 더하는 \(ϵ\) 없음.**  
  `invert_sqrt` 직전에 `add_plain`으로 작은 상수를 넣는 단계가 없다.
- 표준 PyTorch식 LayerNorm과 완전히 동일한 수식을 쓰려면 **`invert_sqrt` 전에 평문 \(ϵ\) 추가**가 필요.

---

## 5. (참고) THOR `layernorm.py` 와의 차이 — 한 줄

- THOR는 **`he_invsqrt`**(적응형 \(k_n\) 루프, 논문의 aSOR 계열 설명과 유사한 취지)와, **조건에 따른 `bootstrap`** 으로 레벨을 회복한다.  
- MOAI 이 파일은 **부트스트랩 없이** Newton+Gold로 `1/√·` 를 처리하는 경로(파라미터 체인이 길게 잡혀 있다는 전제).

---

## 6. Depth / 비용을 줄이는 방향 (실무 체크리스트)

### 6.1 같은 알고리즘 안에서

1. **`invert_sqrt`의 `d_newt`, `d_gold` 축소**  
   수렴·정확도 검증(복호화 또는 plaintext 대조)과 함께 튜닝.
2. **`initGuess` 계수 \((a,b)\) 재피팅**  
   `var` 분포에 맞으면 Newton 횟수를 줄일 수 있음.
3. **Newton만 또는 Goldschmidt만**  
   하이브리드가 두 쪽 depth를 합치므로, 한쪽으로 수렴이 충분하면 다른 쪽 제거 검토.

### 6.2 `invert_sqrt` 자체를 바꿀 때

4. **구간 \([a,b]\)에서 \(1/\sqrt{x}\) 다항식 근사 (Chebyshev / 미니맥스)**  
   - **고정 차수·고정 depth** 설계에 유리.  
   - **입력 구간을 틀리면 오차 폭발** → 대표 데이터에서 **`var`(또는 동일 스케일의 값) min/max·분위수 프로파일링**이 사실상 필요.  
   - 레이어마다 범위가 다르면 **레이어별 구간** 또는 **piecewise** 검토.
5. **`ϵ` 도입**  
   분모 하한을 보장해 \(1/\sqrt{x}\) 구간 설계가 쉬워짐 (표준 LN과도 정합).

### 6.3 연산 구조 / 모델 측

6. **RMSNorm 등 단순화** (모델이 허용할 때만).  
7. **더 긴 모듈러스 체인**  
   “depth를 줄인다”기보다 한 레이어를 통과 가능하게 만드는 쪽.  
8. **부트스트랩**  
   MOAI 현재 이 경로에는 없음; 도입 시 레벨 회복은 되지만 **비용·구현 복잡도** 큼.

### 6.4 다른 파일과의 구분

- **GELU** 다항식 평가는 `gelu_other.cuh` / `doc/gelu_hmult_bsgs.md` 처럼 **BSGS·PS**로 HMult를 줄이는 전략이 직접 해당.  
- **LayerNorm의 병목**은 지금 구조에서는 **`invert_sqrt` + 분산 제곱·출력 ct–ct 곱** 쪽이 중심.

---

## 7. 다음에 하면 좋은 작업 (권장 순서)

1. 실데이터/시뮬로 **`var` 복호 값 또는 동일 정의의 plaintext** 분포 수집.  
2. \(ϵ\) 유무·크기를 정책으로 결정.  
3. (4) 다항식 근사 vs (1–3) 반복 횟수 축소 중 비용·정확도 트레이드오프 비교.

---

*문서 성격: 대화·코드 리딩 기반 요약. 파라미터(스케일, 슬롯 수 768)는 하드코딩 전제를 둔다.*
