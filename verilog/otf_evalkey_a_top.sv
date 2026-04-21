`include "otf_keygen_pkg.sv"
`include "shake128_xof_block.sv"
`include "block_word_issuer.sv"

module otf_evalkey_a_top #(
  parameter int unsigned COEFF_W = otf_keygen_pkg::COEFF_W,
  parameter int unsigned CHUNK_SIZE = otf_keygen_pkg::CHUNK_SIZE,
  parameter int unsigned NUM_SHAKE_LANES = otf_keygen_pkg::NUM_SHAKE_LANES,
  parameter int unsigned NUM_SAMPLER_LANES = otf_keygen_pkg::NUM_SAMPLER_LANES
) (
  input  logic clk,
  input  logic rst_n,

  // Request interface
  input  logic                   req_valid,
  output logic                   req_ready,
  input  otf_keygen_pkg::prng_req_t req,

  // Vector coeff output (one lane per sampler)
  output logic [NUM_SAMPLER_LANES-1:0]                 coeff_valid,
  input  logic [NUM_SAMPLER_LANES-1:0]                 coeff_ready,
  output logic [NUM_SAMPLER_LANES-1:0][COEFF_W-1:0]     coeff_data
);
  import otf_keygen_pkg::*;

  // Sample RTL (parameterizable lanes):
  // - one request at a time, but uses ALL SHAKE and sampler lanes concurrently for that request.
  // - SHAKE lanes output 1344-bit blocks, and each lane issues multiple 64-bit words per cycle.
  prng_req_t cur;
  logic      have_req;
  logic [63:0] remaining;

  localparam int unsigned ISSUE_W_PER_SHAKE = (NUM_SHAKE_LANES == 0) ? 1 : (NUM_SAMPLER_LANES / NUM_SHAKE_LANES);
  localparam int unsigned TOTAL_ISSUE = NUM_SHAKE_LANES * ISSUE_W_PER_SHAKE;
  initial begin
    if (NUM_SHAKE_LANES == 0) $error("NUM_SHAKE_LANES must be >= 1");
    if (NUM_SAMPLER_LANES == 0) $error("NUM_SAMPLER_LANES must be >= 1");
    if ((NUM_SAMPLER_LANES % NUM_SHAKE_LANES) != 0) $error("NUM_SAMPLER_LANES must be divisible by NUM_SHAKE_LANES");
    if (TOTAL_ISSUE != NUM_SAMPLER_LANES) $error("Mapping requires TOTAL_ISSUE == NUM_SAMPLER_LANES");
  end

  // Seed blob per SHAKE lane
  logic [NUM_SHAKE_LANES-1:0] seed_valid_l, seed_ready_l;
  logic [NUM_SHAKE_LANES-1:0][511:0] seed_bits_l;
  logic [NUM_SHAKE_LANES-1:0][15:0]  seed_bytes_l;

  // SHAKE lanes (block output) + multiword issuer (per lane)
  logic [NUM_SHAKE_LANES-1:0] shake_busy_l;
  logic [NUM_SHAKE_LANES-1:0] squeeze_en_l;

  logic [NUM_SHAKE_LANES-1:0] blk_valid_l, blk_ready_l;
  logic [NUM_SHAKE_LANES-1:0][1343:0] blk_data_l;

  logic [NUM_SHAKE_LANES-1:0][ISSUE_W_PER_SHAKE-1:0] iss_valid_l;
  logic [NUM_SHAKE_LANES-1:0][ISSUE_W_PER_SHAKE-1:0] iss_ready_l;
  logic [NUM_SHAKE_LANES-1:0][ISSUE_W_PER_SHAKE-1:0][63:0] iss_word_l;

  genvar gi;
  generate
    for (gi = 0; gi < NUM_SHAKE_LANES; gi++) begin : g_shake
      shake128_xof_block #(.SEED_W(512)) u_shake_blk (
        .clk(clk), .rst_n(rst_n),
        .seed_valid(seed_valid_l[gi]),
        .seed_ready(seed_ready_l[gi]),
        .seed_bits(seed_bits_l[gi]),
        .seed_bytes(seed_bytes_l[gi]),
        .squeeze_en(squeeze_en_l[gi]),
        .block_valid(blk_valid_l[gi]),
        .block_ready(blk_ready_l[gi]),
        .block_data(blk_data_l[gi]),
        .busy(shake_busy_l[gi])
      );

      // Issue multiple 64b words per cycle from 1344b blocks.
      block_word_issuer #(
        .WORD_W(64),
        .WORDS_PER_BLOCK(21),
        .ISSUE_W(ISSUE_W_PER_SHAKE),
        .BLOCK_FIFO_DEPTH( (BITFIFO_DEPTH_WORDS/21 < 2) ? 2 : (BITFIFO_DEPTH_WORDS/21) )
      ) u_issuer (
        .clk(clk), .rst_n(rst_n),
        .in_block_valid(blk_valid_l[gi]),
        .in_block_ready(blk_ready_l[gi]),
        .in_block_data(blk_data_l[gi]),
        .out_valid(iss_valid_l[gi]),
        .out_ready(iss_ready_l[gi]),
        .out_word(iss_word_l[gi])
      );

      // Generate blocks for all lanes while request is active.
      // Backpressure is handled by block_ready from issuer's internal FIFO.
      assign squeeze_en_l[gi] = have_req;
    end
  endgenerate

  // Sampler lanes (parameterized). Feed from issued candidate words (static mapping).
  logic [NUM_SAMPLER_LANES-1:0] s_in_valid_l, s_in_ready_l;
  logic [NUM_SAMPLER_LANES-1:0][63:0] s_in_word_l;
  logic [NUM_SAMPLER_LANES-1:0] s_out_valid_l, s_out_ready_l;
  logic [NUM_SAMPLER_LANES-1:0][COEFF_W-1:0] s_out_coeff_l;
  logic [NUM_SAMPLER_LANES-1:0][31:0] stat_words_l, stat_acc_l, stat_rej_l;

  generate
    for (gi = 0; gi < NUM_SAMPLER_LANES; gi++) begin : g_samp
      reject_sampler_lane u_sampler (
        .clk(clk), .rst_n(rst_n),
        .q(cur.q),
        .threshold_T(cur.threshold_T),
        .in_valid(s_in_valid_l[gi]),
        .in_ready(s_in_ready_l[gi]),
        .in_word(s_in_word_l[gi]),
        .out_valid(s_out_valid_l[gi]),
        .out_ready(s_out_ready_l[gi]),
        .out_coeff(s_out_coeff_l[gi][63:0]),
        .stat_words(stat_words_l[gi]),
        .stat_accepts(stat_acc_l[gi]),
        .stat_rejects(stat_rej_l[gi])
      );
    end
  endgenerate

  // Static mapping issuer slots -> sampler lanes:
  //   sampler i gets from shake lane (i / ISSUE_W_PER_SHAKE), slot (i % ISSUE_W_PER_SHAKE).
  generate
    for (gi = 0; gi < NUM_SAMPLER_LANES; gi++) begin : g_map
      localparam int unsigned SHI = (gi / ISSUE_W_PER_SHAKE);
      localparam int unsigned SLO = (gi % ISSUE_W_PER_SHAKE);
      assign s_in_valid_l[gi] = have_req && iss_valid_l[SHI][SLO];
      assign s_in_word_l[gi]  = iss_word_l[SHI][SLO];
      assign iss_ready_l[SHI][SLO] = s_in_ready_l[gi];
    end
  endgenerate

  // Vector output mapping
  assign coeff_valid = s_out_valid_l;
  assign coeff_data  = s_out_coeff_l;
  always_comb begin
    for (int si=0; si<NUM_SAMPLER_LANES; si++) begin
      s_out_ready_l[si] = have_req && coeff_ready[si];
    end
  end

  // Request handling
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      have_req <= 1'b0;
      cur <= '0;
      remaining <= '0;
      req_ready <= 1'b0;
      for (int li=0; li<NUM_SHAKE_LANES; li++) begin
        seed_valid_l[li] <= 1'b0;
        seed_bits_l[li]  <= '0;
        seed_bytes_l[li] <= 16'd0;
      end
    end else begin
      for (int li = 0; li < NUM_SHAKE_LANES; li++) seed_valid_l[li] <= 1'b0;

      // Accept new request only when idle and ALL SHAKE lanes are ready for a new seed.
      req_ready <= (!have_req) && (&seed_ready_l);
      if (req_valid && req_ready) begin
        cur <= req;
        remaining <= req.num_coeffs;
        have_req <= 1'b1;

        // Seed packing per lane for domain separation (sample layout):
        // [master_seed(256)] [key_id(64)] [decomp_id(64)] [limb_id(64)] [poly_id^lane(64)]
        for (int li=0; li<NUM_SHAKE_LANES; li++) begin
          seed_bits_l[li]  <= { (req.poly_id ^ 64'(li)), req.limb_id, req.decomp_id, req.key_id, req.master_seed[255:0] };
          seed_bytes_l[li] <= 16'd64;
          seed_valid_l[li] <= 1'b1;
        end
      end

      // Decrement remaining by number of accepted outputs this cycle.
      if (have_req && (remaining != 0)) begin
        int unsigned dec;
        dec = 0;
        for (int si=0; si<NUM_SAMPLER_LANES; si++) begin
          if (s_out_valid_l[si] && s_out_ready_l[si]) dec++;
        end
        if (dec != 0) begin
          if (remaining <= 64'(dec)) begin
            remaining <= 64'd0;
            have_req <= 1'b0;
          end else begin
            remaining <= remaining - 64'(dec);
          end
        end
      end
    end
  end

endmodule

