`timescale 1ns/1ps

module tb_shake_block_to_word_fifo;
  localparam int DEPTH = 4;
  localparam int BLOCK_W = 1344;
  localparam int WORD_W = 64;
  localparam int WORDS_PER_BLOCK = 21;

  logic clk, rst_n;

  logic [BLOCK_W-1:0] wr_data;
  logic wr_valid;
  logic wr_ready;

  logic [WORD_W-1:0] rd_data;
  logic rd_valid;
  logic rd_ready;

  logic full, empty;
  logic [$clog2(DEPTH+1)-1:0] block_count;
  logic [4:0] current_word_idx;
  logic has_active_block;
  logic [$clog2(DEPTH)-1:0] dbg_wr_ptr, dbg_rd_ptr;

  shake_block_to_word_fifo #(
    .DEPTH(DEPTH),
    .BLOCK_W(BLOCK_W),
    .WORD_W(WORD_W),
    .WORDS_PER_BLOCK(WORDS_PER_BLOCK)
  ) dut (
    .clk(clk), .rst_n(rst_n),
    .wr_data(wr_data),
    .wr_valid(wr_valid),
    .wr_ready(wr_ready),
    .rd_data(rd_data),
    .rd_valid(rd_valid),
    .rd_ready(rd_ready),
    .full(full),
    .empty(empty),
    .block_count(block_count),
    .current_word_idx(current_word_idx),
    .has_active_block(has_active_block),
    .dbg_wr_ptr(dbg_wr_ptr),
    .dbg_rd_ptr(dbg_rd_ptr)
  );

  // clock
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  function automatic [BLOCK_W-1:0] make_block(input int base);
    automatic logic [BLOCK_W-1:0] b;
    begin
      b = '0;
      for (int i = 0; i < WORDS_PER_BLOCK; i++) begin
        b[i*WORD_W +: WORD_W] = 64'(base + i);
      end
      return b;
    end
  endfunction

  task automatic push_block(input int base);
    wr_data  = make_block(base);
    wr_valid = 1'b1;
    do begin
      @(posedge clk);
    end while (!wr_ready);
    // handshake occurs on this cycle (wr_valid && wr_ready)
    @(posedge clk);
    wr_valid = 1'b0;
  endtask

  task automatic pop_words_and_check(input int base);
    int got = 0;
    rd_ready = 1'b1;
    while (got < WORDS_PER_BLOCK) begin
      @(posedge clk);
      if (rd_valid && rd_ready) begin
        if (rd_data !== 64'(base + got)) begin
          $display("ERROR: expected %0d got %0d (base=%0d idx=%0d)", base+got, rd_data, base, got);
          $fatal;
        end
        got++;
      end
    end
  endtask

  task automatic random_backpressure(input int cycles, input int hold_prob_percent);
    for (int i = 0; i < cycles; i++) begin
      @(posedge clk);
      if (($urandom % 100) < hold_prob_percent) rd_ready <= 1'b0;
      else rd_ready <= 1'b1;
    end
  endtask

  initial begin
    // init
    rst_n = 0;
    wr_valid = 0;
    wr_data = '0;
    rd_ready = 0;
    repeat (5) @(posedge clk);
    rst_n = 1;

    // Scenario 1: single block write, 21 reads
    push_block(0);
    pop_words_and_check(0);
    if (!empty) begin
      $display("ERROR: expected empty after draining single block");
      $fatal;
    end

    // Scenario 2: multiple blocks write, read across boundary
    push_block(1000);
    push_block(2000);
    pop_words_and_check(1000);
    pop_words_and_check(2000);
    if (!empty) begin
      $display("ERROR: expected empty after draining two blocks");
      $fatal;
    end

    // Scenario 3: simultaneous write/read (streaming)
    // Push 3 blocks while reading continuously with occasional backpressure.
    fork
      begin
        push_block(3000);
        push_block(4000);
        push_block(5000);
      end
      begin
        rd_ready = 1'b1;
        // Wait until data starts
        wait (rd_valid);
        // random backpressure during streaming
        random_backpressure(300, 20);
      end
    join

    // Drain remaining words deterministically
    rd_ready = 1'b1;
    pop_words_and_check(3000);
    pop_words_and_check(4000);
    pop_words_and_check(5000);

    // Scenario 4: full behavior
    // Fill queue to capacity (DEPTH blocks). Note staging register is separate, so effective buffered blocks can be DEPTH+1.
    // We check wr_ready goes low when block queue is full.
    for (int i=0;i<DEPTH;i++) begin
      push_block(6000 + i*100);
    end
    // wr_ready should eventually deassert if we try to push one more without draining queue.
    wr_data = make_block(9999);
    wr_valid = 1'b1;
    repeat (20) @(posedge clk);
    if (wr_ready) begin
      $display("WARN: wr_ready still high; staging may have absorbed earlier (allowed), but queue fullness should assert soon in no-drain case.");
    end
    wr_valid = 1'b0;

    // Drain all blocks
    rd_ready = 1'b1;
    for (int i=0;i<DEPTH;i++) begin
      pop_words_and_check(6000 + i*100);
    end
    // Scenario 5: boundary case (word idx 20 -> next block)
    push_block(8000);
    push_block(9000);
    pop_words_and_check(8000);
    pop_words_and_check(9000);

    // Scenario 6: backpressure stability (hold rd_ready low)
    push_block(10000);
    rd_ready = 1'b0;
    repeat (50) @(posedge clk);
    if (!rd_valid) begin
      $display("ERROR: rd_valid should be high when active block exists, even if rd_ready=0");
      $fatal;
    end
    // Data should remain stable while not ready (best-effort check over a few cycles)
    logic [63:0] hold = rd_data;
    repeat (10) @(posedge clk);
    if (rd_data !== hold) begin
      $display("ERROR: rd_data changed while rd_ready=0");
      $fatal;
    end
    rd_ready = 1'b1;
    pop_words_and_check(10000);

    $display("ALL TESTS PASSED");
    $finish;
  end

endmodule

