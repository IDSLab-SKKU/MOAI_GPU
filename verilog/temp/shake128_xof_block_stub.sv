// Sample-only SHAKE128 XOF "block" lane stub.
// Generates a deterministic stream of 1344-bit blocks (21x64b words) from a descriptor.
// This is NOT cryptographically correct. It is for datapath / backpressure / throughput wiring.
module shake128_xof_block_stub #(
  parameter int unsigned DESC_W = 640,
  parameter int unsigned WORD_W = 64,
  parameter int unsigned WORDS_PER_BLOCK = 21,
  parameter int unsigned BLOCK_W = WORDS_PER_BLOCK*WORD_W,
  parameter int unsigned STARTUP_CYCLES = 8,
  parameter int unsigned BLOCK_CYCLES = 4
) (
  input  logic                 clk,
  input  logic                 rst_n,

  // Start a new stream. Once started, blocks keep coming.
  input  logic                 start,
  input  logic [DESC_W-1:0]    descriptor,

  output logic                 block_valid,
  input  logic                 block_ready,
  output logic [BLOCK_W-1:0]   block_data
);
  typedef enum logic [1:0] {IDLE, STARTUP, RUN} state_t;
  state_t st;

  logic [31:0] startup_cnt;
  logic [31:0] block_cnt;

  // Simple 64-bit xorshift* PRNG state (sample-only).
  logic [WORD_W-1:0] s;

  function automatic logic [WORD_W-1:0] fold_desc64(input logic [DESC_W-1:0] d);
    int k;
    logic [WORD_W-1:0] acc;
    begin
      acc = 64'h9E3779B97F4A7C15;
      for (k = 0; k < DESC_W; k += WORD_W) begin
        acc ^= d[k +: WORD_W];
        acc = {acc[62:0], acc[63]} ^ (acc >> 7);
      end
      return acc ^ 64'hD1B54A32D192ED03;
    end
  endfunction

  function automatic logic [WORD_W-1:0] xorshift64star(input logic [WORD_W-1:0] x);
    logic [WORD_W-1:0] y;
    begin
      y = x;
      y ^= (y >> 12);
      y ^= (y << 25);
      y ^= (y >> 27);
      return y * 64'h2545F4914F6CDD1D;
    end
  endfunction

  // Block storage (stable while block_valid=1).
  logic [BLOCK_W-1:0] blk;

  // Produce the next block deterministically from internal state and block counter.
  task automatic gen_block();
    int w;
    logic [WORD_W-1:0] t;
    begin
      t = s ^ (64'(block_cnt) * 64'hA24BAED4963EE407);
      for (w = 0; w < WORDS_PER_BLOCK; w++) begin
        t = xorshift64star(t + 64'(w));
        blk[w*WORD_W +: WORD_W] = t;
      end
      s = xorshift64star(t ^ 64'h0F1E2D3C4B5A6978);
    end
  endtask

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= IDLE;
      startup_cnt <= 0;
      block_cnt <= 0;
      s <= 64'h1;
      block_valid <= 1'b0;
      blk <= '0;
    end else begin
      case (st)
        IDLE: begin
          block_valid <= 1'b0;
          if (start) begin
            s <= fold_desc64(descriptor);
            startup_cnt <= 0;
            block_cnt <= 0;
            st <= STARTUP;
          end
        end

        STARTUP: begin
          block_valid <= 1'b0;
          if (startup_cnt == STARTUP_CYCLES-1) begin
            st <= RUN;
            block_cnt <= 0;
          end else begin
            startup_cnt <= startup_cnt + 1;
          end
        end

        RUN: begin
          // Hold block_valid until accepted.
          if (block_valid && block_ready) begin
            block_valid <= 1'b0;
          end

          // When not holding a valid block, emit a new one every BLOCK_CYCLES cycles.
          if (!block_valid) begin
            if (block_cnt % BLOCK_CYCLES == 0) begin
              gen_block();
              block_valid <= 1'b1;
            end
            block_cnt <= block_cnt + 1;
          end
        end
      endcase
    end
  end

  assign block_data = blk;
endmodule

