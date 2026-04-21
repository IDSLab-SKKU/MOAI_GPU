## `otf_keygen_pkg.sv` 설명

이 파일은 OTF eval-key `a` 생성 RTL 샘플에서 **공통 파라미터와 데이터 타입**을 정의하는 패키지입니다.  
다른 모듈들이 서로 같은 폭/구조를 공유할 수 있게 “한 곳에 모아둔 헤더” 역할을 합니다.

---

## 1) 파라미터(설계의 기본 상수)

```sv
parameter int unsigned COEFF_W = 64;
parameter int unsigned CHUNK_SIZE = 32;
```

- **`COEFF_W`**: 출력 coeff의 비트폭. 샘플은 편의상 u64(64b)를 씁니다.
  - 실제 하드웨어에서는 \(q_j\) 비트폭(예: 50~60b)에 맞춰 줄일 수 있습니다.
- **`CHUNK_SIZE`**: AXI-Stream 한 beat에 실어 보내는 coeff 개수.
  - downstream(evk mult)가 스트리밍으로 처리하기 쉽게 “묶음 단위”를 정의합니다.

```sv
parameter int unsigned NUM_SHAKE_LANES = 4;
parameter int unsigned NUM_SAMPLER_LANES = 4;
```

- 샘플 구조상 **확장 가능성**을 보여주기 위해 “lane” 파라미터를 둡니다.
- 현재 `otf_evalkey_a_top.sv` 구현은 단순화를 위해 **lane0만 실제 사용**합니다.
  - 문서/구조 측면에서만 multi-lane을 염두에 둔 상태입니다.

```sv
parameter int unsigned BITFIFO_DEPTH_WORDS = 256;
parameter int unsigned COEFF_FIFO_DEPTH   = 1024;
```

- **`BITFIFO_DEPTH_WORDS`**: SHAKE에서 나온 64-bit word를 저장할 FIFO 깊이(각 lane 기준).
- **`COEFF_FIFO_DEPTH`**: coeff를 저장할 FIFO 깊이(샘플 top에서는 coeff FIFO를 생략하고 chunker에 직결).

---

## 2) Request descriptor 타입: `prng_req_t`

```sv
typedef struct packed {
  logic [255:0] master_seed;
  logic [63:0]  key_id;
  logic [63:0]  decomp_id;
  logic [63:0]  limb_id;
  logic [63:0]  poly_id;
  logic [63:0]  q;
  logic [63:0]  threshold_T;
  logic [63:0]  num_coeffs;
} prng_req_t;
```

이 구조체는 “한 번의 생성 요청”을 표현합니다.

- **`master_seed`**: PRNG의 루트 seed. 같은 seed면 같은 결과가 나오도록 합니다(결정론).
- **`key_id/decomp_id/limb_id/poly_id`**: **domain separation**용 메타데이터.
  - 서로 다른 키/디컴포지션/limb/폴리 조합이 같은 `master_seed`를 공유해도 출력 스트림이 섞이지 않게,
    SHAKE absorb 입력에 함께 넣습니다.
- **`q`**: 현재 limb modulus \(q_j\). reject sampler는 `0 <= coeff < q`를 만들기 위해 이것을 사용합니다.
- **`threshold_T`**: reject sampling의 임계값 \(T=\lfloor 2^{64}/q \rfloor q\).
  - **샘플 RTL 단순화 옵션**입니다.
  - `threshold_T != 0`이면 하드웨어가 `T`를 계산할 필요 없이 `x < T` 비교만 하면 됩니다.
  - `threshold_T == 0`이면 샘플 RTL에서 느린 `/` 연산으로 `T`를 계산합니다(가독성용).
- **`num_coeffs`**: 생성해야 하는 coeff 개수.

---

## 3) AXI-Stream chunk payload 타입: `axis_chunk_t`

```sv
typedef struct packed {
  logic [CHUNK_SIZE*COEFF_W-1:0] data;
  logic                          last;
} axis_chunk_t;
```

- **`data`**: coeff들을 붙여 만든 덩어리(예: 32×64b = 2048b).
  - index i의 coeff는 일반적으로 `data[i*COEFF_W +: COEFF_W]`로 꺼냅니다.
- **`last`**: 현재 request의 마지막 chunk인지 표시합니다.
  - AXI-Stream 관례로는 `tlast` 신호로 나가며, consumer가 request boundary를 인지할 수 있게 합니다.

---

## 4) 이 패키지가 전체 설계에서 연결되는 방식

1. `otf_evalkey_a_top.sv`가 `prng_req_t`를 받아 SHAKE seed로 직렬화/absorb합니다.
2. `reject_sampler_lane.sv`가 `q`와 `threshold_T`를 사용해 accept/reject 및 `x % q`를 수행합니다.
3. `chunk_assembler.sv`가 `CHUNK_SIZE`에 맞춰 coeff들을 모아 `tdata/tlast` 형태로 내보냅니다.

