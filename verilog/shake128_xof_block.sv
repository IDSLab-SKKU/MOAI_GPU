// SHAKE128 XOF wrapper that outputs one full SHAKE128 rate block per handshake.
// - Internally uses shake128_xof (64-bit word stream).
// - Accumulates 21x64b words into a 1344-bit block.
//
// Interface:
// - block_valid/block_ready with block_data[1343:0]
// - block ordering matches word ordering:
//   word0 -> bits[63:0], ..., word20 -> bits[1343:1280]
module shake128_xof_block #(
  parameter int unsigned SEED_W = 512,
  parameter int unsigned WORD_W = 64,
  parameter int unsigned WORDS_PER_BLOCK = 21,
  parameter int unsigned BLOCK_W = WORD_W * WORDS_PER_BLOCK
) (
  input  logic              clk,
  input  logic              rst_n,

  input  logic              seed_valid,
  output logic              seed_ready,
  input  logic [SEED_W-1:0] seed_bits,
  input  logic [15:0]       seed_bytes,

  input  logic              squeeze_en,

  output logic              block_valid,
  input  logic              block_ready,
  output logic [BLOCK_W-1:0] block_data,

  output logic              busy
);
  logic w_valid, w_ready;
  logic [WORD_W-1:0] w_data;

  shake128_xof #(.SEED_W(SEED_W)) u_xof (
    .clk(clk), .rst_n(rst_n),
    .seed_valid(seed_valid),
    .seed_ready(seed_ready),
    .seed_bits(seed_bits),
    .seed_bytes(seed_bytes),
    .squeeze_en(squeeze_en),
    .word_valid(w_valid),
    .word_ready(w_ready),
    .word_data(w_data),
    .busy(busy)
  );

  logic [$clog2(WORDS_PER_BLOCK+1)-1:0] idx;
  logic [BLOCK_W-1:0] buf;

  // We can accept words when we're assembling and not holding a completed block.
  assign w_ready = (!block_valid) && (squeeze_en);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      idx <= '0;
      buf <= '0;
      block_valid <= 1'b0;
      block_data <= '0;
    end else begin
      // If a completed block is waiting and consumer accepts it, clear and restart accumulation.
      if (block_valid && block_ready) begin
        block_valid <= 1'b0;
        block_data <= '0;
        idx <= '0;
        buf <= '0;
      end

      // Accumulate words into buffer
      if (w_valid && w_ready) begin
        buf[idx*WORD_W +: WORD_W] <= w_data;
        if (idx == WORDS_PER_BLOCK-1) begin
          // Completed one 1344b block; present it.
          block_valid <= 1'b1;
          block_data <= buf; // note: last word written via nonblocking; good enough for sample RTL
          idx <= idx; // hold
        end else begin
          idx <= idx + 1;
        end
      end

      // When reaching last word, ensure block_data includes it (sample fix-up)
      if (w_valid && w_ready && (idx == WORDS_PER_BLOCK-1)) begin
        block_data[idx*WORD_W +: WORD_W] <= w_data;
      end
    end
  end

endmodule

