// Reject sampler lane:
// - Input: 64-bit candidate words + modulus q + threshold_T (optional)
// - Output: 64-bit coeff = x % q when x < T
//
// Notes:
// - If threshold_T == 0, this sample uses a slow (synthesizable) divide to compute T = floor(2^64/q)*q.
// - x % q uses % operator (slow). For real hardware: replace with Barrett/Montgomery.
module reject_sampler_lane (
  input  logic        clk,
  input  logic        rst_n,

  input  logic [63:0] q,
  input  logic [63:0] threshold_T,

  input  logic        in_valid,
  output logic        in_ready,
  input  logic [63:0] in_word,

  output logic        out_valid,
  input  logic        out_ready,
  output logic [63:0] out_coeff,

  output logic [31:0] stat_words,
  output logic [31:0] stat_accepts,
  output logic [31:0] stat_rejects
);
  logic [63:0] T_eff;
  logic [127:0] two64;
  logic [127:0] k;

  // Compute T = floor(2^64/q)*q (slow) if threshold_T not provided.
  always_comb begin
    if (threshold_T != 0) begin
      T_eff = threshold_T;
    end else if (q != 0) begin
      two64 = (128'h1 << 64);
      k = two64 / q; // slow divide
      T_eff = (k * q)[63:0];
    end else begin
      T_eff = 64'h0;
    end
  end

  // Simple 1-stage accept/reject
  logic accept;
  logic [63:0] coeff_calc;

  always_comb begin
    accept = (in_word < T_eff);
    coeff_calc = (q == 0) ? 64'h0 : (in_word % q);
  end

  // Handshake: only consume input when we can produce output in same cycle or output is idle.
  assign in_ready  = out_ready || !out_valid;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_valid <= 1'b0;
      out_coeff <= '0;
      stat_words <= '0;
      stat_accepts <= '0;
      stat_rejects <= '0;
    end else begin
      if (in_valid && in_ready) begin
        stat_words <= stat_words + 1;
        if (accept) begin
          stat_accepts <= stat_accepts + 1;
          out_valid <= 1'b1;
          out_coeff <= coeff_calc;
        end else begin
          stat_rejects <= stat_rejects + 1;
          // no output
          out_valid <= 1'b0;
        end
      end else if (out_valid && out_ready) begin
        out_valid <= 1'b0;
      end
    end
  end

endmodule

