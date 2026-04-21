## `shake128_xof.sv` 설명 (SHAKE128 XOF lane)

이 모듈은 SHAKE128을 “한 lane”으로 구현해, **64-bit word 스트림**을 생성합니다.

SHAKE128은 Keccak-f[1600] permutation을 기반으로 하는 XOF이며,

- seed(메타데이터)를 absorb해 state를 초기화하고
- padding을 넣어 finalize한 뒤
- 원하는 만큼 squeeze해서 출력 스트림을 생성

하는 구조입니다.

---

## 1) SHAKE128 파라미터: rate = 168 bytes

SHAKE128의 rate는 168B(=1344 bits) 입니다. 즉,

- 한 번의 permutation 결과에서 **처음 168B**가 출력/흡수에 사용되고
- 더 많은 출력이 필요하면 permutation을 다시 돌려 다음 블록을 얻습니다.

RTL에서는 168B를 64-bit로 쪼개 **21개의 64-bit word**로 다룹니다:

- `RATE_WORDS = 168/8 = 21`
- `word_idx`가 0..20을 순회하며 `word_data`를 출력

---

## 2) 인터페이스

### seed 입력
- `seed_valid`, `seed_ready`
- `seed_bits[SEED_W-1:0]`
- `seed_bytes`: seed_bits에서 유효한 byte 수

샘플 구현은 seed를 “고정 길이 blob”로 받고, 이를 **1 byte/cycle**로 absorb 합니다.

### squeeze 출력(64-bit word 스트림)
- `squeeze_en`: 1이면 출력 스트림을 생산(“요청 중” 또는 “FIFO 여유 있음” 등에 연결)
- `word_valid`, `word_ready`, `word_data[63:0]`

샘플 구현은 단순화를 위해 `word_ready`가 1일 때만 `word_valid`를 올리며 word를 내보냅니다.
(즉, downstream이 막히면 word 생산이 중단되는 backpressure 친화 형태)

---

## 3) 내부 상태기계(st)

```sv
typedef enum logic [2:0] {IDLE, ABSORB, PERM0, FINALIZE, SQUEEZE, PERM_SQ} st_t;
```

### `IDLE`
- state를 0으로 초기화
- `seed_valid`가 들어오면 `ABSORB`로 이동

### `ABSORB`
- `absorb_pos`를 0부터 증가시키며
- `seed_bits`의 byte를 state의 rate 영역에 XOR

핵심 함수:
- `xor_byte(state, byte_off, b)`:
  - `byte_off`가 속한 lane/byte offset을 계산해 state에 XOR

### padding(중요)
SHAKE padding 규칙(샘플):
- 현재 pos에 `0x1F` XOR
- rate의 마지막 바이트에 `0x80` XOR

함수 `apply_shake_pad(state, pos)`가 이걸 수행합니다.

### `PERM0` → `FINALIZE`
- `keccak_f1600`을 `start`하여 permutation 1회를 수행
- `perm_done`이 오면 `state <= perm_out`로 갱신 후 `SQUEEZE`로 이동

### `SQUEEZE`
`squeeze_en`이 1일 때:
- `word_idx` 위치의 lane을 `word_data`로 출력
- `word_idx`가 20이면 rate 블록을 다 쓴 것이므로 `PERM_SQ`로 이동

### `PERM_SQ`
다음 출력 블록을 위해 permutation을 다시 돌립니다.
완료되면 state 갱신 후 다시 `SQUEEZE`.

---

## 4) `word_data`가 state에서 나오는 방식

```sv
word_data <= get_lane(state, word_idx);
```

즉, rate 영역의 lane(0..20)을 순서대로 64-bit씩 뽑습니다.

이 출력 스트림이 downstream FIFO(`sync_fifo`)로 들어가고,
그 FIFO가 reject sampler에 의해 소비됩니다.

---

## 5) 이 모듈과 reject sampling의 연결 관점

reject sampler는 “필요한 만큼”의 64-bit 후보 word를 소비합니다.

- accept가 잘 되면(대부분 \(p\\approx 1\)) word 소비량 ≈ coeff 생성량
- reject가 많아지면 word 소비량이 증가 → SHAKE lane에 더 많은 squeeze 요구

그래서 설계에서는

- SHAKE가 빠르게 word를 뽑아줄 수 있고
- 중간의 bit/word FIFO가 순간적인 소비 증가(reject burst)를 흡수하며
- consumer backpressure가 upstream에 전달

되도록 구성합니다.

