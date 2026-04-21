`timescale 1ns/1ps

module tb_otf_evalkey_a_top_vec;
  import otf_keygen_pkg::*;

  // Use package defaults (NUM_SHAKE_LANES=2, NUM_SAMPLER_LANES=8 after recent update)
  localparam int unsigned NL = NUM_SAMPLER_LANES;

  logic clk, rst_n;

  logic req_valid, req_ready;
  prng_req_t req;

  logic [NL-1:0]             coeff_valid;
  logic [NL-1:0]             coeff_ready;
  logic [NL-1:0][COEFF_W-1:0] coeff_data;

  otf_evalkey_a_top dut (
    .clk(clk), .rst_n(rst_n),
    .req_valid(req_valid),
    .req_ready(req_ready),
    .req(req),
    .coeff_valid(coeff_valid),
    .coeff_ready(coeff_ready),
    .coeff_data(coeff_data)
  );

  // clock
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  task automatic do_reset();
    begin
      rst_n = 0;
      req_valid = 0;
      coeff_ready = '0;
      req = '0;
      repeat (5) @(posedge clk);
      rst_n = 1;
      repeat (2) @(posedge clk);
    end
  endtask

  // Random-ish ready pattern (deterministic)
  int unsigned rng;
  function automatic logic randbit();
    begin
      // xorshift32
      rng ^= (rng << 13);
      rng ^= (rng >> 17);
      rng ^= (rng <<  5);
      return rng[0];
    end
  endfunction

  // Capture first N accepted coeffs per lane for determinism checks
  localparam int unsigned N_CAP = 32;
  logic [NL-1:0][N_CAP-1:0][COEFF_W-1:0] capA, capB;
  int unsigned capcnt [NL-1:0];

  task automatic clear_caps();
    for (int i=0;i<NL;i++) capcnt[i] = 0;
  endtask

  // Run one request, consume until "done" (req_ready returns high), and capture samples.
  task automatic run_one(
    input  logic [255:0] master_seed,
    input  logic [63:0]  key_id,
    input  logic [63:0]  decomp_id,
    input  logic [63:0]  limb_id,
    input  logic [63:0]  poly_id,
    input  logic [63:0]  q,
    input  logic [63:0]  threshold_T,
    input  logic [63:0]  num_coeffs,
    input  bit           enable_backpressure,
    output int unsigned  total_accepted,
    output logic [NL-1:0][N_CAP-1:0][COEFF_W-1:0] caps
  );
    int unsigned cycles;
    begin
      total_accepted = 0;
      cycles = 0;
      clear_caps();

      // Program request
      req.master_seed  = master_seed;
      req.key_id       = key_id;
      req.decomp_id    = decomp_id;
      req.limb_id      = limb_id;
      req.poly_id      = poly_id;
      req.q            = q;
      req.threshold_T  = threshold_T;
      req.num_coeffs   = num_coeffs;

      // Start with ready high (or random if backpressure)
      coeff_ready = '1;
      if (enable_backpressure) begin
        // Start from deterministic seed
        rng = 32'hC001D00D;
      end

      // Issue req_valid until accepted
      @(posedge clk);
      req_valid = 1;
      while (!req_ready) @(posedge clk);
      @(posedge clk);
      req_valid = 0;

      // Consume until done: top goes idle => req_ready=1 (and we don't assert req_valid)
      while (1) begin
        @(posedge clk);
        cycles++;

        if (enable_backpressure) begin
          for (int i=0;i<NL;i++) begin
            // 75% chance ready=1
            coeff_ready[i] = randbit() | 1'b1;
            coeff_ready[i] = (rng[3:0] != 4'h0);
          end
        end else begin
          coeff_ready = '1;
        end

        for (int i=0;i<NL;i++) begin
          if (coeff_valid[i] && coeff_ready[i]) begin
            total_accepted++;
            // Range check: coeff < q (when q != 0)
            if (q != 0) begin
              if (!(coeff_data[i] < q)) begin
                $display("ERROR: coeff_data[%0d]=%0d not < q=%0d", i, coeff_data[i], q);
                $fatal;
              end
            end
            if (capcnt[i] < N_CAP) begin
              caps[i][capcnt[i]] = coeff_data[i];
              capcnt[i] = capcnt[i] + 1;
            end
          end
        end

        // Done condition: core idle (req_ready high) and we are not trying to send a request.
        if (req_ready) begin
          if (total_accepted != num_coeffs) begin
            $display("ERROR: accepted=%0d != requested=%0d", total_accepted, num_coeffs);
            $fatal;
          end
          break;
        end

        if (cycles > 200000) begin
          $display("TIMEOUT: cycles=%0d accepted=%0d", cycles, total_accepted);
          $fatal;
        end
      end
    end
  endtask

  // Precompute threshold_T = floor(2^64/q)*q in TB to avoid slow divider in reject_sampler_lane.
  function automatic logic [63:0] compute_threshold_T(input logic [63:0] q);
    logic [127:0] two64;
    logic [127:0] k;
    logic [127:0] T;
    begin
      if (q == 0) begin
        return 64'd0;
      end
      two64 = (128'h1 << 64);
      k = two64 / q;
      T = k * q;
      return T[63:0];
    end
  endfunction

  initial begin
    do_reset();

    // Choose q. We precompute threshold_T in TB for faster simulation.
    logic [63:0] q;
    logic [63:0] threshold_T;
    q = 64'd12289;
    threshold_T = compute_threshold_T(q);

    int unsigned acc0, acc1;

    // Run A: no backpressure, capture caps
    run_one(
      256'h00010203_04050607_08090A0B_0C0D0E0F_10111213_14151617_18191A1B_1C1D1E1F,
      64'd1, 64'd2, 64'd3, 64'd4,
      q, threshold_T, 64'd256,
      0,
      acc0,
      capA
    );

    // Reset and run B: identical request, enable backpressure, compare determinism of accepted sequence per lane.
    do_reset();
    run_one(
      256'h00010203_04050607_08090A0B_0C0D0E0F_10111213_14151617_18191A1B_1C1D1E1F,
      64'd1, 64'd2, 64'd3, 64'd4,
      q, threshold_T, 64'd256,
      1,
      acc1,
      capB
    );

    // Determinism check (best-effort): compare first N_CAP captured values per lane.
    for (int i=0;i<NL;i++) begin
      for (int j=0;j<N_CAP;j++) begin
        if (capA[i][j] !== capB[i][j]) begin
          $display("ERROR: determinism mismatch lane=%0d idx=%0d A=%h B=%h", i, j, capA[i][j], capB[i][j]);
          $fatal;
        end
      end
    end

    $display("tb_otf_evalkey_a_top_vec: PASS");
    #20;
    $finish;
  end

endmodule

