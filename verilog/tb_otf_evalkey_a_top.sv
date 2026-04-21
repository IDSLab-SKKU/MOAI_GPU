`timescale 1ns/1ps

module tb_otf_evalkey_a_top;
  import otf_keygen_pkg::*;

  logic clk, rst_n;

  logic req_valid, req_ready;
  prng_req_t req;

  logic [NUM_SAMPLER_LANES-1:0]              coeff_valid;
  logic [NUM_SAMPLER_LANES-1:0]              coeff_ready;
  logic [NUM_SAMPLER_LANES-1:0][COEFF_W-1:0]  coeff_data;

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

  // simple stimulus
  initial begin
    rst_n = 0;
    req_valid = 0;
    coeff_ready = '1;
    req = '0;
    repeat (5) @(posedge clk);
    rst_n = 1;

    // Request: generate 64 coeffs for q=12289, provide threshold_T=0 (compute in RTL)
    @(posedge clk);
    req.master_seed = 256'h00010203_04050607_08090A0B_0C0D0E0F_10111213_14151617_18191A1B_1C1D1E1F;
    req.key_id = 64'd1;
    req.decomp_id = 64'd2;
    req.limb_id = 64'd3;
    req.poly_id = 64'd4;
    req.q = 64'd12289;
    // Precompute threshold_T in TB to avoid slow divider in sampler
    begin
      logic [127:0] two64, k, T;
      two64 = (128'h1 << 64);
      k = two64 / req.q;
      T = k * req.q;
      req.threshold_T = T[63:0];
    end
    req.num_coeffs = 64'd64;
    req_valid = 1;
    while (!req_ready) @(posedge clk);
    @(posedge clk);
    req_valid = 0;

    // Consume output, add backpressure for a few cycles
    int accepts = 0;
    int cycles = 0;
    while (1) begin
      @(posedge clk);
      cycles++;
      if (cycles == 50) coeff_ready <= '0;
      if (cycles == 80) coeff_ready <= '1;

      for (int i=0;i<NUM_SAMPLER_LANES;i++) begin
        if (coeff_valid[i] && coeff_ready[i]) begin
          accepts++;
          if (coeff_data[i] >= req.q) begin
            $display("ERROR: coeff[%0d]=%0d >= q=%0d", i, coeff_data[i], req.q);
            $fatal;
          end
        end
      end

      if (req_ready) begin
        if (accepts != req.num_coeffs) begin
          $display("ERROR: accepts=%0d != req.num_coeffs=%0d", accepts, req.num_coeffs);
          $fatal;
        end
        $display("DONE: accepts=%0d cycles=%0d", accepts, cycles);
        #20;
        $finish;
      end

      if (cycles > 20000) begin
        $display("TIMEOUT");
        $fatal;
      end
    end
  end

endmodule

