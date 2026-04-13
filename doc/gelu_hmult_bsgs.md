## GELU 근사(`gelu_v2`)에서 CT×CT(HMult) 횟수와 BSGS 적용 제안

- **대상 파일**: `MOAI_GPU/src/include/source/non_linear_func/gelu_other.cuh`
- **대상 함수**: `PhantomCiphertext gelu_v2(...)`

### 결론 요약

- **현재 구현의 CT×CT(HMult) 총 횟수**: **23회**
  - **Square(CT×CT)**: 4회
  - **Multiply(CT×CT)**: 19회
- **`gelu_v2` 한 번 호출 시 depth(레벨 소비) — `gelu_other.cuh` 코드 기준**:
  - **`rescale_to_next_inplace` 총합**: **48회**  
    입력 `multiply_plain` 1회 + `square`/`multiply`로 `x^1..x^24` 생성 시 각 HMult 직후 23회 + 계수 곱 루프 `i=1..24`에서 `multiply_plain`마다 24회. 상수항은 `add_plain`만(rescale 없음).
  - **CT×CT만 세는 다항식 곱셈 깊이(임계 경로)**: **5**  
    스케일된 `x^1`의 ct–ct 깊이를 0으로 두면 `x^2→x^4→x^8→x^16`이 square 4번으로 깊이 4이고, `x^17=x^1·x^16` 등 한 번 더 multiply로 **최대 5**.
  - **주의**: 계수 루프의 24번 `multiply_plain`도 각각 `rescale`을 호출하므로 **체인에서 소비하는 rescale 횟수**는 **48**이 맞고, “ct–ct 깊이 5”와는 별개 지표다. (표는 아래 §1.3.)
- **BSGS/Paterson–Stockmeyer(PS) 적용 시 기대**:
  - `x^1..x^24`를 거의 전부 만드는 방식 대신, 블록 분해 후 필요한 거듭제곱만 생성
  - **HMult를 11회 수준**으로 줄이는 스케줄이 가능(아래 제안)

---

## 1) CT×CT(HMult) 카운트 (코드 기준)

`gelu_v2()`에서 CT×CT 곱이 발생하는 지점은 아래 두 종류뿐이다.

- `evaluator.square(...)`  → **CT×CT**
- `evaluator.multiply(...)` → **CT×CT**

반면 `evaluator.multiply_plain_*` 는 **CT×PT**이므로 본 문서의 HMult 카운트에는 포함하지 않는다.

### 1.1 Square(CT×CT) = 4회

다음 거듭제곱을 만들기 위해 총 4번 수행:

- `x^2`
- `x^4`
- `x^8`
- `x^16`

### 1.2 Multiply(CT×CT) = 19회

아래 루프/연산에서 총 19번 수행:

- **x^3, x^5, x^9, x^17** 만들기: 4회  
  - `for (i=2; i<17; i*=2)` → i=2,4,8,16
- **x^6, x^10, x^18** 만들기: 3회  
  - `for (i=4; i<17; i*=2)` → i=4,8,16
- **x^7, x^11, x^19** 만들기: 3회  
  - `for (i=4; i<17; i*=2)` → i=4,8,16
- **x^12, x^20** 만들기: 2회  
  - `for (i=8; i<17; i*=2)` → i=8,16
- **x^13, x^21** 만들기: 2회
- **x^14, x^22** 만들기: 2회
- **x^15, x^23** 만들기: 2회
- **x^24 = x^8 * x^16**: 1회

따라서 총합:

- **HMult = 4 (square) + 19 (multiply) = 23회**

### 1.3 `rescale_to_next` 횟수와 “depth” 정의 (보충)

CKKS/Phantom 계열에서 `rescale_to_next` 한 번은 보통 모듈러스 체인을 한 단계 내린다. `gelu_v2`에서는 거의 모든 `square` / `multiply` / 계수용 `multiply_plain` 직후에 `rescale_to_next_inplace`가 따라온다(`gelu_other.cuh` `L37–38`, `L49–139`, `L160–175`).

| 구간 | 내용 | `rescale` 횟수 |
|------|------|----------------|
| 스케일 | `x * 0.1` (`multiply_plain`) | 1 |
| `x^1..x^24` 생성 | 각 `square` / `multiply` 직후 | 23 |
| \(\sum_{i=1}^{24} a_i x^i\) | 각 `multiply_plain` 직후 (`i=1..24`) | 24 |
| 상수항 | `add_plain` | 0 |
| **합계** | | **48** |

**CT×CT 다항식 깊이**만 따로 셀 때는, 거듭제곱 DAG 상 **임계 경로 = square 4 + multiply 1 = 5**(위 §결론 요약과 동일).

---

## 2) 왜 HMult가 큰가?

현재 구현은 degree 24 다항식 \(P(x)\)를 계산할 때,

- 먼저 `x^i (i=1..24)`를 다수 생성 (여기서 HMult가 대부분 발생)
- 그 다음 \(\sum a_i x^i\)는 각 항마다 `multiply_plain + add`로 누적 (여기는 CT×PT 중심)

즉 **“평가 자체”는 CT×PT로 비교적 싸게 하고, “거듭제곱 테이블 생성”이 비싼 구조**다.

---

## 3) BSGS/Paterson–Stockmeyer(PS)로 줄이는 방법 (블록 크기 \(m=5\))

degree 24 다항식

\[
P(x)=\sum_{k=0}^{24} a_k x^k
\]

을 다음처럼 블록으로 묶는다.

\[
P(x)=Q_0(x) + x^5 Q_1(x) + x^{10} Q_2(x) + x^{15} Q_3(x) + x^{20} Q_4(x)
\]

여기서 각 \(Q_j(x)\)는 degree < 5:

\[
Q_j(x) = b_{j,0} + b_{j,1}x + b_{j,2}x^2 + b_{j,3}x^3 + b_{j,4}x^4
\]

따라서 \(Q_j\)는 **baby power** \(\{x,x^2,x^3,x^4\}\)만 있으면 `multiply_plain`과 `add`로 구성 가능(= HMult 없이 생성 가능)하고,
큰 비용은 \(x^{5j}\)와 \(Q_j\)의 CT×CT 결합에만 집중된다.

---

## 4) 제안 스케줄 (HMult = 11)

아래에서 HMult는 **CT×CT multiply/square**만 카운트한다. (실제로는 보통 각 HMult 이후 `relinearize + rescale`이 뒤따름)

### 4.1 Baby powers 만들기: 4 HMult

- **(1)** \(x^2 = x \square\)  → HMult 1
- **(2)** \(x^4 = (x^2) \square\) → HMult 2
- **(3)** \(x^3 = x^2 \cdot x\) → HMult 3
- **(4)** \(x^5 = x^4 \cdot x\) → HMult 4  
  - (대안) \(x^5=x^3\cdot x^2\)도 가능. 레벨/스케일 맞추기 쉬운 쪽을 선택.

### 4.2 Giant powers 만들기: 3 HMult (누적 7)

- **(5)** \(x^{10} = (x^5)\square\) → HMult 5
- **(6)** \(x^{15} = x^{10}\cdot x^5\) → HMult 6
- **(7)** \(x^{20} = (x^{10})\square\) → HMult 7

### 4.3 블록 \(Q_0..Q_4\) 만들기: HMult 0

각 \(Q_j\)는 아래처럼 구성:

\[
Q_j = b_{j,0} + b_{j,1}x + b_{j,2}x^2 + b_{j,3}x^3 + b_{j,4}x^4
\]

구현상 각 항은 `multiply_plain` + `rescale`(+ 필요 시 `mod_switch_to`) 후 누적 `add`로 처리.

### 4.4 최종 결합: 4 HMult (총 11)

- **(8)** \(T_1 = x^5 \cdot Q_1\) → HMult 8
- **(9)** \(T_2 = x^{10}\cdot Q_2\) → HMult 9
- **(10)** \(T_3 = x^{15}\cdot Q_3\) → HMult 10
- **(11)** \(T_4 = x^{20}\cdot Q_4\) → HMult 11

마지막으로

\[
P(x)=Q_0 + T_1 + T_2 + T_3 + T_4
\]

는 add만 수행하면 된다.

---

## 5) 계수 인덱스 매핑 (현재 코드 배열 → BSGS 블록)

### 5.1 현재 코드의 계수 의미

- 다항식 \(P(x)=\sum_{k=0}^{24} a_k x^k\)
- 코드의 `coeff_high_to_low[]`는 고차항부터 저장되어 있으며,
  - \[
    a_k = \texttt{coeff\_high\_to\_low}[24-k]
    \]
- 코드에서 \(x^i\)에 곱하는 계수는 `coeff_high_to_low[24 - i]`
- 상수항 \(a_0\)는 마지막에 `coeff_high_to_low[24]`를 더하는 방식이다.

### 5.2 BSGS(\(m=5\)) 분해에서의 매핑

\[
Q_j(x)=\sum_{r=0}^{4} b_{j,r} x^r,\quad b_{j,r}=a_{5j+r}
\]

따라서 배열 인덱스 기준으로

\[
b_{j,r} = \texttt{coeff\_high\_to\_low}\big[\,24-(5j+r)\,\big]
\]

#### 블록별 인덱스 표 (바로 구현용)

- \(Q_0\): \([24,23,22,21,20]\) → \((a_0..a_4)\)
- \(Q_1\): \([19,18,17,16,15]\) → \((a_5..a_9)\)
- \(Q_2\): \([14,13,12,11,10]\) → \((a_{10}..a_{14})\)
- \(Q_3\): \([9,8,7,6,5]\) → \((a_{15}..a_{19})\)
- \(Q_4\): \([4,3,2,1,0]\) → \((a_{20}..a_{24})\)

---

## 6) HMult 감소가 성능에 미치는 영향(기대)

- CKKS에서 보통 가장 비싼 연산은 **CT×CT 곱(square/multiply) + relinearize + rescale**.
- 현재 구현은 HMult가 23회로 거듭제곱 생성 비용이 크다.
- BSGS/PS 스케줄로 11회 수준까지 줄이면,
  - 곱셈/키스위칭/리스케일 비중이 큰 경우 **대체로 유의미한 속도 향상**이 기대된다.
- 단, 실제 속도는 `mod_switch` 빈도, 파라미터 체인(레벨 수), 구현의 커널 최적화 정도에 따라 달라진다.

