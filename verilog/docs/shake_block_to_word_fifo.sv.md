## `shake_block_to_word_fifo.sv` 설명 (1344b block → 64b word width-converting FIFO)

이 모듈은 **SHAKE128의 출력 블록(1344 bits = 168B)**을 “블록 단위로 enqueue”하고,  
이를 **64-bit word 스트림(21개)**으로 “점진적으로 dequeue”하는 **폭 변환 FIFO/버퍼**입니다.

표준 same-width FIFO와 다른 점:
- write는 “블록 1개”가 원자적 단위
- read는 “워드 1개”가 원자적 단위
- **occupancy는 block 단위로만** 관리
- 내부적으로는
  - 블록 큐(`DEPTH`개)
  - current block staging 레지스터
  - word index(0..20)
  를 둬서 progressive slicing을 수행합니다.

---

## 1) 외부 인터페이스

### Write side (block enqueue)
- `wr_data[1343:0]`
- `wr_valid`
- `wr_ready`

**enqueue 조건**: `wr_valid && wr_ready`  
→ 블록 큐에 1344b 블록 1개 저장

### Read side (word dequeue)
- `rd_data[63:0]`
- `rd_valid`
- `rd_ready`

**dequeue 조건**: `rd_valid && rd_ready`  
→ current block의 현재 word를 1개 소비하고, word index 증가

### Status
- `full`: 블록 큐가 가득 찼음(DEPTH blocks)
- `empty`: (블록 큐 비었음) AND (current block도 없음)
- `block_count`: 큐에 쌓인 블록 수(스테이징 레지스터는 포함하지 않음)

### Debug(선택)
- `current_word_idx`: 0..20
- `has_active_block`: current staging 유효 여부
- `dbg_wr_ptr/dbg_rd_ptr`: 큐 포인터

---

## 2) word ordering(중요)

요구사항대로:
- word0 = `wr_data[63:0]`
- word1 = `wr_data[127:64]`
- ...
- word20 = `wr_data[1343:1280]`

즉 slice는:

\[
rd\\_data = cur\\_block[word\\_idx*64 +: 64]
\]

---

## 3) 내부 상태(레지스터)

### Block queue (DEPTH blocks)
- `mem[0:DEPTH-1]` : 1344b block 저장
- `wptr/rptr` : block 단위 포인터
- `count` : 큐에 들어있는 block 개수

### Current block staging
- `cur_block` : 현재 읽고 있는 1344b block
- `cur_valid` : staging에 block이 들어있는지
- `word_idx` : 0..20 (현재 word 위치)

중요: **staging은 큐 용량과 분리**되어 있어, `block_count`에 포함되지 않습니다.

---

## 4) Load-next-block policy

정책:
- current block이 없고(`cur_valid=0`)
- 큐에 block이 있으면(`count!=0`)

→ 큐에서 block 1개를 pop하여 staging에 로드하고 `word_idx=0`으로 시작합니다.

이 로드는
- reset 직후 첫 read가 필요해졌을 때
- 또는 이전 block의 word20을 소비해 staging이 비는 순간
자동으로 수행됩니다.

---

## 5) empty/full 정의(요구사항 반영)

- `full` = (큐 `count == DEPTH`)
  - staging은 capacity에 포함하지 않음
- `empty` = (큐 `count == 0`) AND (`cur_valid == 0`)
  - 큐가 비어도 staging에 word가 남아있으면 empty가 아님

---

## 6) 동시에 read/write 되는 경우

이 구조는 다음을 안전하게 처리하도록 설계됐습니다:
- 같은 cycle에 enqueue(do_wr)와 current word dequeue(do_rd)가 동시에 일어날 수 있음
- word20 소비로 staging이 비는 cycle에, 큐에서 즉시 다음 block을 staging으로 로드할 수 있음(정책상 “간단하고 robust”하게 처리)

---

## 7) TB

동봉 TB: `tb_shake_block_to_word_fifo.sv`
- 단일 block → 21 word 순서 검사
- 다중 block → 경계 통과 검사
- read/write 동시 + backpressure 랜덤
- full/empty behavior
- rd_ready=0일 때 rd_data 안정성(간단 check)

