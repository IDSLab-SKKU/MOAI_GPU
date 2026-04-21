## OTF eval-key `a` keygen 엔진 블록다이어그램 & 읽는 순서

요청한 “그림(PNG)”은 현재 환경에서 이미지 생성 기능과 `graphviz(dot)`가 모두 없어 **자동 생성은 불가**합니다.  
대신 아래에 **블록다이어그램(mermaid)**을 넣어두었고, 이걸 그대로 복사해서 GitHub/Notion/Markdown mermaid 렌더러에서 보면 “그림”처럼 확인할 수 있습니다.

---

## 1) 전체 블록다이어그램 (데이터플로 + backpressure)

```mermaid
flowchart LR
  Req[ReqInterface\n(seed+ids+q+T+num_coeffs)] --> DescQ[DescriptorQueue\n(샘플 top은 1개 in-flight)]
  DescQ --> DomainSep[DomainSeparation\nSeedSerialize/Absorb]
  DomainSep --> Shake[SHAKE128_XOF\n(keccak_f1600)]
  Shake --> WordFifo[WordFIFO\n(64-bit words)]
  WordFifo --> Sampler[RejectSampler\n(x<T ? x%q : reject)]
  Sampler --> CoeffFifo[CoeffFIFO\n(샘플 top은 생략 가능)]
  CoeffFifo --> Chunker[ChunkAssembler\n(CHUNK_SIZE coeffs)]
  Chunker --> AXIS[AXI-Stream Out\n(tvalid/tready/tdata/tlast)]
  AXIS --> Consumer[evkMult Consumer]

  Consumer -. backpressure .-> AXIS
  AXIS -. backpressure .-> Chunker
  Chunker -. backpressure .-> CoeffFifo
  CoeffFifo -. backpressure .-> Sampler
  Sampler -. backpressure .-> WordFifo
  WordFifo -. backpressure .-> Shake
```

### 노트(직관용)
- **SHAKE128 rate=168B** → permutation 블록당 **21×64b word**를 squeeze
- **reject sampling**: `x<T`만 accept해서 `x%q` 출력(모듈러 bias 제거)
- 출력 coeff는 설계상 “**NTT-domain `a`**”로 태그되어 downstream이 추가 NTT 없이 소비(개념적 가정)

---

## 2) 어떤 순서로 문서를 보면 되는지(추천)

PRNG 원리를 잘 모르는 상태에서 “위에서 아래로” 이해하기 좋은 순서입니다.

1. **전체 큰 그림**: `verilog/README_otf_keygen_rtl.md`
2. **타입/요청 구조**: `verilog/docs/otf_keygen_pkg.sv.md`
3. **SHAKE 출력이 왜 필요한지(64-bit 후보 스트림)**: `verilog/docs/shake128_xof.sv.md`
4. **SHAKE 내부 핵심(Keccak permutation)**: `verilog/docs/keccak_f1600.sv.md`
5. **균일 샘플링 핵심(reject sampling)**: `verilog/docs/reject_sampler_lane.sv.md`
6. **chunk 스트리밍/AXI-Stream backpressure**: `verilog/docs/chunk_assembler.sv.md`
7. **모든 연결(top)**: `verilog/docs/otf_evalkey_a_top.sv.md`
8. **TB로 전체 흐름 확인**: `verilog/docs/tb_otf_evalkey_a_top.sv.md`

---

## 3) “진짜 PNG 그림”이 꼭 필요하면

아래 중 하나로 만들 수 있습니다.

- **옵션 A (권장)**: 로컬/서버에 `graphviz` 설치 후, mermaid를 dot/png로 변환하거나 draw.io 사용
- **옵션 B**: VSCode/Notion/GitHub에서 mermaid 렌더링으로 그대로 사용

