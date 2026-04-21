## `keccak_f1600.sv` 설명 (Keccak-f[1600] permutation)

이 모듈은 SHAKE128의 핵심인 **Keccak-f[1600] permutation**을 수행합니다.

- 입력: 1600-bit state
- 출력: 1600-bit state'
- 총 24 rounds를 순차적으로 수행(샘플 RTL: 1 round/cycle 형태로 단순화)

> 이 구현은 “맞게 돌아가는 최소 구현”에 가깝고, 실제 하드웨어에서는 파이프라이닝/언롤/비트슬라이스 등으로 최적화합니다.

---

## 1) Keccak state 표현

Keccak 상태는 5×5 배열의 64-bit lane으로 표현됩니다.

- lane index 관례: `lane = x + 5*y`
- RTL에서는 `logic [63:0] A[0:4][0:4];`로 접근합니다.

`state_in[1599:0]`는 아래 순서로 unpack 됩니다:

```sv
A[x][y] = s[64*(x+5*y) +: 64];
```

그리고 최종 결과는 같은 방식으로 pack 합니다.

---

## 2) 라운드 함수 구성

Keccak-f 라운드는 다음 순서로 진행됩니다.

1. **Theta**
2. **Rho**
3. **Pi**
4. **Chi**
5. **Iota**

### 2.1 Theta

컬럼 parity를 구합니다:

\[
C[x] = A[x,0] \oplus A[x,1] \oplus A[x,2] \oplus A[x,3] \oplus A[x,4]
\]

그리고

\[
D[x] = C[x-1] \oplus \text{rot}(C[x+1], 1)
\]

를 만들어 모든 row에 XOR 합니다:

\[
A[x,y] \leftarrow A[x,y] \oplus D[x]
\]

RTL에서는 `C[0..4]`, `D[0..4]`로 계산합니다.

### 2.2 Rho + Pi

- **Rho**: 각 lane을 고정된 offset만큼 회전(rotate-left)
- **Pi**: lane 위치를 재배치

샘플 RTL에서는 Rho offset을 `rho(x,y)` 함수로 제공하고,
Pi 매핑은 구현 내의

```sv
int nx = y;
int ny = (2*x + 3*y) % 5;
B[nx][ny] = rol64(A[x][y], rho(x,y));
```

로 합쳐서 수행합니다.

### 2.3 Chi

비선형 단계로 각 row에서:

\[
A[x,y] \leftarrow B[x,y] \oplus ((\neg B[x+1,y]) \wedge B[x+2,y])
\]

를 적용합니다.

### 2.4 Iota

라운드 상수 `RC[round]`를 `A[0][0]`에 XOR 합니다.

---

## 3) 핸드셰이크/상태기계

### 입력/출력 신호
- `start`: 1이면 `state_in`을 래치하고 연산 시작
- `done`: 결과가 `state_out`에 유효한 1-cycle 펄스

### 상태
- `IDLE`: `start` 대기. 들어오면 unpack 후 `RUN`으로.
- `RUN`: `round`를 0..23 진행하며 라운드 수행.
- `OUT`: pack 후 `done=1`, 다시 `IDLE`.

---

## 4) 샘플 구현의 한계(중요)

- 이 구현은 “아키텍처 설명용”이며, 실제로는 다음 최적화가 필요합니다.
  - round 언롤 또는 파이프라이닝
  - lane 연산 병렬화/공유 로직 최적화
  - `always_ff` 내 nonblocking 할당과 임시 배열(B 등)의 조합을 더 명확하게 정리 (합성/시뮬레이터 호환)


