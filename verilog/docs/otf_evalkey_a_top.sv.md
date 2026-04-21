## `otf_evalkey_a_top.sv` 설명 (Top-level 연결)

이 모듈은 OTF eval-key `a` 생성 엔진의 **탑 레벨 샘플**입니다.

구성은 아래 데이터플로를 그대로 연결합니다:

1) request 수신 (`req_valid/req_ready`)  
2) SHAKE seed absorb → SHAKE 64-bit word 스트림 생성  
3) word FIFO에 저장  
4) reject sampler가 word를 소비하여 coeff 생성  
5) chunk assembler가 coeff를 `CHUNK_SIZE` 단위로 AXI-Stream 출력  

> 샘플 구현은 “이해용”이므로 **한 번에 1 request만 처리**하고, **lane0만 사용**합니다.

---

## 1) 인터페이스

### Request 입력
- `req_valid`, `req_ready`
- `req` (`otf_keygen_pkg::prng_req_t`)

### 출력(AXI-Stream 스타일)
- `m_axis_tvalid`, `m_axis_tready`
- `m_axis_tdata = CHUNK_SIZE*COEFF_W` 비트
- `m_axis_tlast`

---

## 2) 내부 레지스터: request 상태 관리

```sv
prng_req_t cur;
logic have_req;
logic [63:0] remaining;
```

- `cur`: 현재 처리 중인 request descriptor
- `have_req`: active request가 있는지
- `remaining`: 아직 생성해야 하는 coeff 개수

`remaining==1`인 coeff를 막 chunker로 넘긴 순간, request가 끝난 것으로 처리합니다.

---

## 3) SHAKE seed 구성(domain separation)

샘플에서는 seed를 “완전 직렬화” 대신 단순 pack으로 만들었습니다:

```sv
seed_bits <= {poly_id, limb_id, decomp_id, key_id, master_seed};
seed_bytes <= 64;
```

의미:
- 같은 `master_seed`라도 `(key_id, decomp_id, limb_id, poly_id)`가 다르면 다른 스트림이 나오도록 함(의도).

> 실제 구현에서는 lane_id, output_format 등의 추가 필드와 “라벨 문자열”을 함께 absorb해서\n+> 더 강한 domain separation을 합니다.

---

## 4) SHAKE → word FIFO 연결(backpressure)

`shake128_xof` 출력은 `sync_fifo`에 push 됩니다.

핵심 연결:

```sv
w_ready   = !wf_full;
wf_wr     = w_valid && w_ready;
squeeze_en = have_req && !wf_full;
```

즉 FIFO가 가득 차면:
- SHAKE의 `word_ready`가 내려가고
- `squeeze_en`도 내려가서 SHAKE가 더 이상 word를 만들지 않게 됩니다.

이게 “생산자(SHAKE)가 소비자(샘플러) 속도에 맞춰 스톨”하는 구조입니다.

---

## 5) word FIFO → reject sampler 연결

```sv
s_in_valid = have_req && !wf_empty;
wf_rd      = s_in_valid && s_in_ready;
```

sampler가 받을 준비가 되면(`s_in_ready=1`) FIFO에서 pop 합니다.

sampler는 accept 시에만 coeff를 출력하므로,
reject가 많아지면 FIFO pop이 늘어나게 되어 SHAKE 소비량이 증가합니다.

---

## 6) reject sampler → chunk assembler 연결

샘플 top은 coeff FIFO를 두지 않고, sampler output을 chunker에 직결했습니다:

```sv
ca_s_valid = s_out_valid;
s_out_ready = ca_s_ready && have_req;
```

chunker가 막히면(`ca_s_ready=0`) sampler도 막히고, FIFO pop도 멈추며, SHAKE도 멈춥니다.

---

## 7) remaining 카운터 및 tlast 생성

```sv
coeff_last = (remaining == 1);
s_last = coeff_last && ca_s_valid && ca_s_ready;
```

즉 마지막 coeff가 실제로 chunker에 accept되는 사이클에 `s_last`가 1이 되고,
chunker는 이를 `m_axis_tlast`로 전파합니다(마지막 chunk일 때).

---

## 8) 이 탑을 확장하려면(실제 설계 방향)

1) **multi-lane**:\n+   - 여러 `shake128_xof`와 여러 `reject_sampler_lane`를 인스턴스화\n+   - request를 lane에 매핑(예: `lane_id = limb_id % NUM_SHAKE_LANES`)\n+   - per-lane FIFO 및 arbitration 필요\n+\n+2) **다중 in-flight request**:\n+   - descriptor queue를 두고, 각 request별 remaining/last 처리를 분리\n+   - chunk output에 request boundary를 태그하거나, request별 AXIS 채널 분리\n+\n+3) **나눗셈 제거**:\n+   - `threshold_T`를 SW에서 계산해 내려주기\n+   - `% q`를 Barrett/Montgomery로 교체\n+\n+4) **coeff FIFO 추가**:\n+   - chunker 앞에 coeff FIFO를 두면 producer/consumer decouple이 좋아짐\n+
