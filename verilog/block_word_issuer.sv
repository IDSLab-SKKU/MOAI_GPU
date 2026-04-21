// Takes 1344-bit blocks (21x64b words) and issues up to ISSUE_W 64-bit words per cycle.
// Intended to bridge "block-based SHAKE lane" to multi-lane samplers.
module block_word_issuer #(
  parameter int unsigned WORD_W = 64,
  parameter int unsigned WORDS_PER_BLOCK = 21,
  parameter int unsigned BLOCK_W = WORDS_PER_BLOCK*WORD_W,
  parameter int unsigned ISSUE_W = 4,
  parameter int unsigned BLOCK_FIFO_DEPTH = 4
) (
  input  logic                   clk,
  input  logic                   rst_n,

  // Input blocks
  input  logic                   in_block_valid,
  output logic                   in_block_ready,
  input  logic [BLOCK_W-1:0]     in_block_data,

  // Issued candidate words (vector)
  output logic [ISSUE_W-1:0]                     out_valid,
  input  logic [ISSUE_W-1:0]                     out_ready,
  output logic [ISSUE_W-1:0][WORD_W-1:0]         out_word
);
  localparam int unsigned PTR_W = (BLOCK_FIFO_DEPTH <= 2) ? 1 : $clog2(BLOCK_FIFO_DEPTH);

  logic [BLOCK_FIFO_DEPTH-1:0][BLOCK_W-1:0] mem;
  logic [PTR_W-1:0] wptr, rptr;
  logic [PTR_W:0]   count;

  // Current block staging
  logic             have_blk;
  logic [BLOCK_W-1:0] cur_blk;
  logic [$clog2(WORDS_PER_BLOCK+1)-1:0] cur_idx;

  // FIFO push/pop
  wire fifo_full  = (count == BLOCK_FIFO_DEPTH);
  wire fifo_empty = (count == 0);

  assign in_block_ready = !fifo_full;

  // Helper to extract word j from a block.
  function automatic logic [WORD_W-1:0] blk_word(input logic [BLOCK_W-1:0] b, input int unsigned j);
    return b[j*WORD_W +: WORD_W];
  endfunction

  // How many issue slots are actually requestable this cycle?
  // We only advance when the slot is both valid and ready.
  // To keep things simple/robust, we require "prefix-ready": if out_ready[k]==0,
  // then slots >k won't be issued (prevents holes and complicated bookkeeping).
  function automatic int unsigned prefix_ready_count(input logic [ISSUE_W-1:0] r);
    int unsigned k;
    begin
      prefix_ready_count = 0;
      for (k=0;k<ISSUE_W;k++) begin
        if (r[k]) prefix_ready_count++;
        else break;
      end
    end
  endfunction

  int unsigned can_take;
  int unsigned avail_words;
  int unsigned take_n;

  always_comb begin
    // Default outputs
    out_valid = '0;
    out_word  = '0;

    // Determine availability
    avail_words = 0;
    if (have_blk) begin
      avail_words = (cur_idx < WORDS_PER_BLOCK) ? (WORDS_PER_BLOCK - cur_idx) : 0;
    end

    // Determine how many we can take this cycle based on ready and availability.
    can_take = prefix_ready_count(out_ready);
    take_n = (avail_words < can_take) ? avail_words : can_take;

    // Assert valids for the first take_n slots, fill words accordingly.
    for (int unsigned k=0;k<ISSUE_W;k++) begin
      if (k < take_n) begin
        out_valid[k] = 1'b1;
        out_word[k]  = blk_word(cur_blk, cur_idx + k);
      end
    end
  end

  // Load new current block from FIFO when needed.
  wire need_blk = (!have_blk) || (have_blk && (cur_idx >= WORDS_PER_BLOCK));

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wptr <= '0;
      rptr <= '0;
      count <= '0;
      have_blk <= 1'b0;
      cur_blk <= '0;
      cur_idx <= '0;
    end else begin
      // Push incoming blocks
      if (in_block_valid && in_block_ready) begin
        mem[wptr] <= in_block_data;
        wptr <= wptr + 1;
        count <= count + 1;
      end

      // If we need a new current block, pop from FIFO
      if (need_blk) begin
        if (!fifo_empty) begin
          cur_blk <= mem[rptr];
          rptr <= rptr + 1;
          count <= count - 1;
          have_blk <= 1'b1;
          cur_idx <= '0;
        end else begin
          have_blk <= 1'b0;
          cur_idx <= '0;
        end
      end else begin
        // Advance index by number of accepted issues (take_n equals accepted due to prefix-ready + out_valid)
        cur_idx <= cur_idx + $clog2(WORDS_PER_BLOCK+1)'(take_n);
      end
    end
  end
endmodule

