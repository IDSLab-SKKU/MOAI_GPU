## `reject_sampler_lane.sv` 설명 (reject sampling lane)

이 모듈은 SHAKE128에서 나온 **64-bit 후보 word 스트림**을 받아서,
각 limb modulus \(q\)에 대해 **균일 분포 coeff**를 생성합니다.

핵심은 **reject sampling**으로 modulo bias를 제거하는 것입니다.

---

## 1) 왜 reject sampling이 필요한가?

단순히 `coeff = x % q`를 하면, `x`가 0..2^64-1에서 균일하더라도
\(q\)가 2의 거듭제곱이 아닌 이상 나머지 분포가 완전히 균일하지 않습니다.

이를 해결하기 위해 임계값 \(T\)를 정의합니다:

\[
T = \left\lfloor \frac{2^{64}}{q} \right\rfloor \cdot q
\]

그리고:
- `x < T`면 accept, 출력 `x % q`
- 아니면 reject하고 다음 `x`를 사용

이러면 `0..q-1`이 균일해집니다.

---

## 2) 인터페이스(스트림 핸드셰이크)

### 입력
- `in_valid`, `in_ready`, `in_word[63:0]`
- `q[63:0]`: modulus
- `threshold_T[63:0]`: (옵션) 미리 계산된 T

### 출력
- `out_valid`, `out_ready`, `out_coeff[63:0]`

즉 전형적인 valid/ready 스트림 1-stage 변환기입니다.

---

## 3) T 계산 방식(샘플의 단순화)

```sv
if (threshold_T != 0) T_eff = threshold_T;
else begin
  k = (2^64) / q;
  T_eff = (k*q)[63:0];
end
```

- `threshold_T != 0`이면 RTL이 나눗셈을 안 해도 됩니다(권장).
- `threshold_T == 0`이면 샘플 RTL에서 `/`로 계산합니다.

> 실제 설계에서는 divider 비용이 크므로, 보통 SW가 `T`를 계산해서 내려주거나,\n+> (q가 고정/소수 집합이면) ROM/LUT로 `T`를 제공하는 것이 현실적입니다.

---

## 4) accept/reject 및 modulo

```sv
accept = (in_word < T_eff);
coeff_calc = (in_word % q);
```

- accept면 coeff 출력
- reject면 출력 없음 (`out_valid=0`)

> `%` 역시 divider가 필요하므로 실제 설계에서는 Barrett/Montgomery로 대체합니다.

---

## 5) backpressure 전파(in_ready)

```sv
in_ready = out_ready || !out_valid;
```

이 의미는:
- 출력이 비어있거나(`!out_valid`)
- 출력이 소비될 준비가 되어 있으면(`out_ready`)

그때만 input을 받아들인다는 뜻입니다.

즉 consumer(뒤쪽)가 막히면 이 lane은 input을 멈추고,
그 backpressure는 upstream FIFO/Shake로 전파될 수 있습니다.

---

## 6) 통계 카운터

- `stat_words`: 입력 후보 word를 몇 번 소비했는지
- `stat_accepts`: accept된 개수(생성된 coeff 개수)
- `stat_rejects`: reject된 개수

이 값으로 대략적인 accept rate을 추정할 수 있습니다:

\[
p \approx \frac{accepts}{words}
\]

---

## 7) 이 모듈을 “진짜 설계”로 바꿀 때

- `threshold_T`는 가능하면 **desc로 전달** (divider 제거)
- `% q`는 **Barrett/Montgomery**로 교체
- reject가 연속으로 발생할 때 처리율을 유지하려면 multi-lane + 충분한 word FIFO 필요

