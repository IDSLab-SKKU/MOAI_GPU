## `chunk_assembler.sv` 설명 (coeff → chunk AXI-Stream 변환)

이 모듈은 **coeff 스트림(1 coeff/beat)** 을 받아서, `CHUNK_SIZE`개씩 모아
**AXI-Stream 스타일 chunk(큰 `tdata`)** 로 내보냅니다.

목적:
- downstream(evk mult)가 “폴리 전체”를 기다리지 않고도 chunk 단위로 처리 가능하게 하기
- `tready` backpressure가 upstream에 자연스럽게 전달되게 하기

---

## 1) 인터페이스

### 입력 coeff 스트림
- `s_valid`, `s_ready`
- `s_coeff[COEFF_W-1:0]`
- `s_last`: 현재 request의 마지막 coeff에 1

### 출력 chunk 스트림 (AXI-Stream 스타일)
- `m_valid`, `m_ready`
- `m_data[CHUNK_SIZE*COEFF_W-1:0]`
- `m_last`: 현재 request의 마지막 chunk 표시(`tlast` 역할)

---

## 2) 내부 상태/버퍼

```sv
logic [$clog2(CHUNK_SIZE+1)-1:0] fill;
logic [CHUNK_SIZE*COEFF_W-1:0] buf;
logic last_seen;
```

- `fill`: 현재 버퍼에 몇 개 coeff가 쌓였는지
- `buf`: coeff들을 순서대로 저장하는 shift-register 역할
- `last_seen`: request의 마지막 coeff(`s_last`)가 들어왔는지 기록

coeff i는 아래 위치에 저장됩니다:

```sv
buf[COEFF_W*fill +: COEFF_W] <= s_coeff;
```

즉 i번째 슬롯이 `i*COEFF_W` 오프셋입니다.

---

## 3) s_ready 조건(핸드셰이크 핵심)

```sv
s_ready = (!m_valid) && (fill < CHUNK_SIZE);
```

의미:
- 이미 출력 chunk가 유효(`m_valid=1`)인데 소비자가 아직 안 가져가면, 입력을 멈춘다.
- 버퍼가 가득 찼는데 출력도 못 하면 입력을 멈춘다.

즉 `m_ready`가 내려가면 곧바로 `s_ready`도 내려가며 upstream을 stall시킵니다.

---

## 4) chunk emit 조건

두 가지 경우에 chunk를 내보냅니다.

### 4.1 full chunk

```sv
if (fill == CHUNK_SIZE) emit
```

### 4.2 last coeff를 받았고, partial chunk라도 남아있을 때

```sv
else if (last_seen && (fill != 0)) emit, m_last=1
```

즉 `num_coeffs`가 `CHUNK_SIZE`의 배수가 아니면 마지막 chunk는 partial이고,
남은 슬롯은 0으로 패딩된 채 `m_data`로 나갑니다(샘플 구현).

---

## 5) backpressure가 전체 시스템에 미치는 영향

이 모듈의 출력이 막히면:

`m_ready=0` → `m_valid`가 유지 → `s_ready=0` → sampler가 coeff를 못 밀어넣음 → word FIFO가 안 빠짐 → SHAKE가 멈춤

즉, chunker는 consumer backpressure를 upstream PRNG 생산까지 전달하는 “댐” 역할을 합니다.

---

## 6) 실설계에서 자주 하는 개선

- `m_valid`를 올린 뒤에도 `m_ready`가 늦게 오는 동안 입력을 계속 받으려면
  - **double-buffer**(ping-pong) 구조로 chunk buffer를 2개 두고\n+  - 한쪽은 output 대기, 다른 쪽은 input 수집\n+  같은 구조를 씁니다.
- partial chunk 패딩 규칙을 consumer와 합의(0-pad vs don't-care)해야 합니다.

