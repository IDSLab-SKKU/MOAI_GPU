`timescale 1ns/1ps

module tb_prng_engine_top;
  localparam int unsigned NUM_SHAKE_LANES = 2;
  localparam int unsigned NUM_SAMPLER_LANES = 8;
  localparam int unsigned ISSUE_W_PER_SHAKE = 4;

  logic clk;
  logic rst_n;

  logic start;
  logic [127:0] domain_tag;
  logic [255:0] master_seed;
  logic [63:0] key_id, decomp_id, limb_id, lane_id_base;
  logic [63:0] q, threshold_T;

  logic [NUM_SAMPLER_LANES-1:0] coeff_valid;
  logic [NUM_SAMPLER_LANES-1:0] coeff_ready;
  logic [NUM_SAMPLER_LANES-1:0][63:0] coeff_data;

  prng_engine_top #(
    .NUM_SHAKE_LANES(NUM_SHAKE_LANES),
    .NUM_SAMPLER_LANES(NUM_SAMPLER_LANES),
    .ISSUE_W_PER_SHAKE(ISSUE_W_PER_SHAKE),
    .STARTUP_CYCLES(4),
    .BLOCK_CYCLES(2),
    .BLOCK_FIFO_DEPTH(2)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .domain_tag(domain_tag),
    .master_seed(master_seed),
    .key_id(key_id),
    .decomp_id(decomp_id),
    .limb_id(limb_id),
    .lane_id_base(lane_id_base),
    .q(q),
    .threshold_T(threshold_T),
    .coeff_valid(coeff_valid),
    .coeff_ready(coeff_ready),
    .coeff_data(coeff_data)
  );

  // Clock
  initial clk = 0;
  always #5 clk = ~clk;

  task automatic reset();
    begin
      rst_n = 0;
      start = 0;
      coeff_ready = '1;
      domain_tag = 128'h0;
      master_seed = 256'h0;
      key_id = 64'h0;
      decomp_id = 64'h0;
      limb_id = 64'h0;
      lane_id_base = 64'h0;
      q = 64'd0;
      threshold_T = 64'd0;
      repeat (5) @(posedge clk);
      rst_n = 1;
      repeat (2) @(posedge clk);
    end
  endtask

  // Capture first N valids per lane for determinism check.
  localparam int unsigned N_CAP = 16;
  logic [NUM_SAMPLER_LANES-1:0][N_CAP-1:0][63:0] cap0;
  logic [NUM_SAMPLER_LANES-1:0][N_CAP-1:0][63:0] cap1;
  int unsigned capcnt [NUM_SAMPLER_LANES-1:0];

  task automatic clear_caps();
    int i;
    begin
      for (i=0;i<NUM_SAMPLER_LANES;i++) begin
        capcnt[i] = 0;
      end
    end
  endtask

  task automatic run_and_capture(output logic [NUM_SAMPLER_LANES-1:0][N_CAP-1:0][63:0] caps);
    int cyc;
    int i;
    begin
      clear_caps();
      for (cyc=0;cyc<400;cyc++) begin
        @(posedge clk);
        for (i=0;i<NUM_SAMPLER_LANES;i++) begin
          if (coeff_valid[i] && coeff_ready[i]) begin
            if (capcnt[i] < N_CAP) begin
              caps[i][capcnt[i]] = coeff_data[i];
              capcnt[i] = capcnt[i] + 1;
            end
            // Range check
            if (q != 0) begin
              if (!(coeff_data[i] < q)) begin
                $display("ERROR: coeff_data[%0d]=%0d not < q=%0d", i, coeff_data[i], q);
                $fatal;
              end
            end
          end
        end
      end
    end
  endtask

  initial begin
    reset();

    // Make rejects likely (~1/2 accept rate): q ~= 2^63+1 gives T=q, accept if x<q.
    q = 64'h8000_0000_0000_0001;
    threshold_T = 64'd0; // let sampler compute T (slow but ok in sim)

    domain_tag   = 128'h4F54465F45564B5F415F50524E475F564543; // "OTF_EVK_A_PRNG_VEC" (truncated)
    master_seed  = 256'h0123456789abcdef_0011223344556677_8899aabbccddeeff_fedcba9876543210;
    key_id       = 64'd7;
    decomp_id    = 64'd2;
    limb_id      = 64'd1;
    lane_id_base = 64'd100;

    // Pulse start and capture
    @(posedge clk);
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;

    run_and_capture(cap0);

    // Reset and repeat same descriptor: deterministic check
    reset();
    q = 64'h8000_0000_0000_0001;
    threshold_T = 64'd0;
    domain_tag   = 128'h4F54465F45564B5F415F50524E475F564543;
    master_seed  = 256'h0123456789abcdef_0011223344556677_8899aabbccddeeff_fedcba9876543210;
    key_id       = 64'd7;
    decomp_id    = 64'd2;
    limb_id      = 64'd1;
    lane_id_base = 64'd100;

    @(posedge clk);
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;

    run_and_capture(cap1);

    // Compare first N_CAP captures per lane (best-effort: if a lane produced fewer, compare those).
    for (int i=0;i<NUM_SAMPLER_LANES;i++) begin
      for (int j=0;j<N_CAP;j++) begin
        if (cap0[i][j] !== cap1[i][j]) begin
          $display("ERROR: determinism mismatch lane=%0d idx=%0d cap0=%h cap1=%h", i, j, cap0[i][j], cap1[i][j]);
          $fatal;
        end
      end
    end

    $display("tb_prng_engine_top: PASS");
    $finish;
  end
endmodule

