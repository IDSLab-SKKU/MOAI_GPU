## `tb_otf_evalkey_a_top.sv` 설명 (샘플 테스트벤치)

이 파일은 `otf_evalkey_a_top`이 최소한의 형태로 동작하는지 확인하기 위한 **간단한 TB**입니다.

목표는 “정확한 암호학 검증”이 아니라:
- request 핸드셰이크가 되는지
- 출력 AXI-Stream chunk가 나오는지
- `tready`를 내려도 backpressure가 걸려 멈췄다가 다시 진행되는지
- 출력 coeff가 `coeff < q` 범위를 벗어나지 않는지(best-effort)

를 빠르게 확인하는 것입니다.

---

## 1) 클럭/리셋

```sv
forever #5 clk = ~clk;
```

- 10ns period → 100MHz

리셋은 몇 cycle 유지 후 해제합니다.

---

## 2) request 생성

TB는 다음 request를 한 번 보냅니다:

- `q = 12289`
- `num_coeffs = 64`
- `threshold_T = 0` (즉, sampler가 RTL 내부에서 나눗셈으로 T 계산)
- seed/ids는 고정 상수

핸드셰이크:

```sv
req_valid = 1;
while (!req_ready) @(posedge clk);
req_valid = 0;
```

---

## 3) 출력 소비 및 backpressure

TB는 기본적으로 `m_axis_tready=1`로 두고 출력을 소비합니다.

중간에 일부 구간에 backpressure를 주기 위해:

```sv
if (cycles == 50) m_axis_tready <= 0;
if (cycles == 80) m_axis_tready <= 1;
```

즉 30 cycle 동안 소비자를 막아,
upstream이 멈추는지(그리고 다시 열면 재개되는지) 확인합니다.

---

## 4) 출력 데이터 검사(best-effort)

출력 beat(`m_axis_tvalid && m_axis_tready`)를 받을 때:

- `CHUNK_SIZE`개의 coeff를 꺼내서\n+  `coeff >= q`면 경고 출력

주의:
- 마지막 chunk는 partial일 수 있고, 샘플 구현은 남는 슬롯을 0으로 남겨둡니다.\n+  따라서 0은 정상 값입니다.\n+
---

## 5) 종료 조건

`m_axis_tlast`가 들어오면:
- request가 끝났다고 보고 `$finish`.

무한 루프 방지를 위해 `cycles > 20000`이면 TIMEOUT으로 종료합니다.

---

## 6) 이 TB를 더 엄밀하게 만들려면

- 동일 seed/ids에 대해 **golden 모델(SW SHAKE128 + reject sampling)**을 돌려\n+  TB에서 실제 coeff 시퀀스를 비교
- domain separation 변화(예: limb_id 1비트 변경) 시 출력이 달라지는지 확인\n+  (확률적이지만 golden과 비교하면 deterministic 검증 가능)
- `threshold_T`를 TB에서 실제로 계산해 넣어 divider 경로를 회피하고,\n+  pure compare/modulo 경로만 검증

