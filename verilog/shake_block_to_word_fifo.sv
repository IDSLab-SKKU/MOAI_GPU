// Width-converting FIFO/buffer:
// - Write side enqueues one SHAKE128 rate block (1344 bits = 21*64b words) per handshake.
// - Read side dequeues 64-bit words progressively from a staged current block.
//
// Key properties:
// - Occupancy tracked in BLOCK units (queue depth = DEPTH blocks).
// - Current block staging register is NOT counted as queue storage.
// - empty is true only when (queue empty) && (no active staged block).
module shake_block_to_word_fifo #(
  parameter int unsigned DEPTH = 4,
  parameter int unsigned BLOCK_W = 1344,
  parameter int unsigned WORD_W  = 64,
  parameter int unsigned WORDS_PER_BLOCK = 21
) (
  input  logic                 clk,
  input  logic                 rst_n,

  // Write side (block enqueue)
  input  logic [BLOCK_W-1:0]   wr_data,
  input  logic                 wr_valid,
  output logic                 wr_ready,

  // Read side (word dequeue)
  output logic [WORD_W-1:0]    rd_data,
  output logic                rd_valid,
  input  logic                rd_ready,

  // Status (block units)
  output logic                 full,
  output logic                 empty,
  output logic [$clog2(DEPTH+1)-1:0] block_count,

  // Debug (nice-to-have)
  output logic [4:0]           current_word_idx,
  output logic                has_active_block,
  output logic [$clog2(DEPTH)-1:0] dbg_wr_ptr,
  output logic [$clog2(DEPTH)-1:0] dbg_rd_ptr
);
  // Sanity: default params match SHAKE128 rate.
  localparam int unsigned AW = (DEPTH <= 1) ? 1 : $clog2(DEPTH);

  // Block queue storage
  logic [BLOCK_W-1:0] mem [0:DEPTH-1];
  logic [AW-1:0] wptr, rptr;
  logic [$clog2(DEPTH+1)-1:0] count;

  // Current block staging
  logic [BLOCK_W-1:0] cur_block;
  logic               cur_valid;
  logic [4:0]         word_idx; // 0..20 for 21 words

  // Handshake internal
  wire do_wr = wr_valid && wr_ready;
  wire do_rd = rd_valid && rd_ready;

  assign full  = (count == DEPTH[$clog2(DEPTH+1)-1:0]);
  assign wr_ready = !full;

  // empty means: no queued blocks and no active staged block
  assign empty = (count == 0) && (!cur_valid);
  assign block_count = count;

  assign has_active_block = cur_valid;
  assign current_word_idx = word_idx;
  assign dbg_wr_ptr = wptr;
  assign dbg_rd_ptr = rptr;

  // rd_valid whenever we have an active block staged
  assign rd_valid = cur_valid;

  // Slice output word: word 0 = [63:0], word 20 = [1343:1280]
  // No padding; BLOCK_W must equal WORD_W * WORDS_PER_BLOCK.
  always_comb begin
    rd_data = '0;
    if (cur_valid) begin
      rd_data = cur_block[word_idx*WORD_W +: WORD_W];
    end
  end

  // Main sequential logic:
  // - Maintain block queue (count/wptr/rptr)
  // - Maintain current block staging (cur_valid/cur_block/word_idx)
  // - Load-next-block policy: if no active block and queue has data, pop into cur_block.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wptr <= '0;
      rptr <= '0;
      count <= '0;
      cur_block <= '0;
      cur_valid <= 1'b0;
      word_idx <= '0;
    end else begin
      // 1) Enqueue block into queue memory
      if (do_wr) begin
        mem[wptr] <= wr_data;
        wptr <= (wptr == DEPTH-1) ? '0 : (wptr + 1);
      end

      // 2) Consume one word from current block (if any)
      if (do_rd) begin
        if (word_idx < (WORDS_PER_BLOCK-1)) begin
          word_idx <= word_idx + 1;
        end else begin
          // Finished last word of this block
          cur_valid <= 1'b0;
          word_idx <= '0;
        end
      end

      // 3) Update queue count based on enqueue and queue-pop-to-staging
      // We'll compute whether we pop from queue into staging this cycle.
      // Policy: after processing rd consumption (which may clear cur_valid),
      // if there is no active block and queue has at least 1 block, load one.
      //
      // Note: We pop from queue memory by advancing rptr and decrementing count.
      // The staging register itself is separate from the queue capacity.
      logic will_load;
      will_load = 1'b0;

      // Determine if cur_valid will be false after rd consumption.
      logic cur_valid_after_rd;
      cur_valid_after_rd = cur_valid;
      if (do_rd && cur_valid) begin
        if (word_idx == (WORDS_PER_BLOCK-1)) cur_valid_after_rd = 1'b0;
      end

      if (!cur_valid_after_rd && (count != 0)) begin
        will_load = 1'b1;
      end

      // Count update includes: +do_wr, -will_load
      unique case ({do_wr, will_load})
        2'b10: count <= count + 1;
        2'b01: count <= count - 1;
        default: count <= count;
      endcase

      // 4) If loading: move one block from queue into cur_block
      if (will_load) begin
        cur_block <= mem[rptr];
        cur_valid <= 1'b1;
        word_idx <= '0;
        rptr <= (rptr == DEPTH-1) ? '0 : (rptr + 1);
      end
    end
  end

endmodule

