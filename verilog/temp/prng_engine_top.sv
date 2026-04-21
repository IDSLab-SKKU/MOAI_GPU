// Full PRNG engine datapath (sample RTL):
// - 2 "SHAKE block" lanes (stubbed) each produce 1344-bit blocks
// - Each block is issued as 4x64b candidate words/cycle (2 lanes -> 8 candidates/cycle)
// - 8 reject sampler lanes convert candidates to coeff mod q
// - Vector output: up to 8 coeffs/cycle with per-lane valid/ready
//
// NOTE: shake128 lanes are stubbed (non-crypto) for datapath illustration.
module prng_engine_top #(
  parameter int unsigned NUM_SHAKE_LANES = 2,
  parameter int unsigned NUM_SAMPLER_LANES = 8,
  parameter int unsigned ISSUE_W_PER_SHAKE = 4, // must satisfy NUM_SHAKE_LANES*ISSUE_W_PER_SHAKE == NUM_SAMPLER_LANES

  parameter int unsigned DESC_W = 640,
  parameter int unsigned WORD_W = 64,
  parameter int unsigned WORDS_PER_BLOCK = 21,
  parameter int unsigned BLOCK_W = WORDS_PER_BLOCK*WORD_W,

  parameter int unsigned STARTUP_CYCLES = 8,
  parameter int unsigned BLOCK_CYCLES = 4,
  parameter int unsigned BLOCK_FIFO_DEPTH = 4
) (
  input  logic clk,
  input  logic rst_n,

  // Request/start
  input  logic                 start,
  input  logic [127:0]         domain_tag,
  input  logic [255:0]         master_seed,
  input  logic [63:0]          key_id,
  input  logic [63:0]          decomp_id,
  input  logic [63:0]          limb_id,
  input  logic [63:0]          lane_id_base,

  // Modulus config
  input  logic [63:0]          q,
  input  logic [63:0]          threshold_T,

  // Vector output
  output logic [NUM_SAMPLER_LANES-1:0]            coeff_valid,
  input  logic [NUM_SAMPLER_LANES-1:0]            coeff_ready,
  output logic [NUM_SAMPLER_LANES-1:0][63:0]      coeff_data
);
  localparam int unsigned TOTAL_ISSUE = NUM_SHAKE_LANES*ISSUE_W_PER_SHAKE;
  initial begin
    if (TOTAL_ISSUE != NUM_SAMPLER_LANES) $error("NUM_SHAKE_LANES*ISSUE_W_PER_SHAKE must equal NUM_SAMPLER_LANES");
  end

  // Pack per-lane descriptors
  logic [NUM_SHAKE_LANES-1:0][DESC_W-1:0] lane_desc;
  prng_descriptor_pack #(
    .DESC_W(DESC_W),
    .NUM_LANES(NUM_SHAKE_LANES)
  ) u_desc (
    .domain_tag(domain_tag),
    .master_seed(master_seed),
    .key_id(key_id),
    .decomp_id(decomp_id),
    .limb_id(limb_id),
    .lane_id_base(lane_id_base),
    .lane_desc(lane_desc)
  );

  // SHAKE block lanes (stubbed)
  logic [NUM_SHAKE_LANES-1:0]           blk_valid;
  logic [NUM_SHAKE_LANES-1:0]           blk_ready;
  logic [NUM_SHAKE_LANES-1:0][BLOCK_W-1:0] blk_data;

  genvar i;
  generate
    for (i=0;i<NUM_SHAKE_LANES;i++) begin : g_shake
      shake128_xof_block_stub #(
        .DESC_W(DESC_W),
        .WORD_W(WORD_W),
        .WORDS_PER_BLOCK(WORDS_PER_BLOCK),
        .STARTUP_CYCLES(STARTUP_CYCLES),
        .BLOCK_CYCLES(BLOCK_CYCLES)
      ) u_lane (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .descriptor(lane_desc[i]),
        .block_valid(blk_valid[i]),
        .block_ready(blk_ready[i]),
        .block_data(blk_data[i])
      );
    end
  endgenerate

  // Block->multiword issuers (4 words/cycle each)
  logic [NUM_SHAKE_LANES-1:0][ISSUE_W_PER_SHAKE-1:0] iss_valid;
  logic [NUM_SHAKE_LANES-1:0][ISSUE_W_PER_SHAKE-1:0] iss_ready;
  logic [NUM_SHAKE_LANES-1:0][ISSUE_W_PER_SHAKE-1:0][WORD_W-1:0] iss_word;

  generate
    for (i=0;i<NUM_SHAKE_LANES;i++) begin : g_iss
      block_word_issuer #(
        .WORD_W(WORD_W),
        .WORDS_PER_BLOCK(WORDS_PER_BLOCK),
        .ISSUE_W(ISSUE_W_PER_SHAKE),
        .BLOCK_FIFO_DEPTH(BLOCK_FIFO_DEPTH)
      ) u_iss (
        .clk(clk),
        .rst_n(rst_n),
        .in_block_valid(blk_valid[i]),
        .in_block_ready(blk_ready[i]),
        .in_block_data(blk_data[i]),
        .out_valid(iss_valid[i]),
        .out_ready(iss_ready[i]),
        .out_word(iss_word[i])
      );
    end
  endgenerate

  // 8 reject sampler lanes
  // Static mapping:
  //   SHAKE lane 0 issue slots [0..3] -> sampler [0..3]
  //   SHAKE lane 1 issue slots [0..3] -> sampler [4..7]
  logic [NUM_SAMPLER_LANES-1:0] samp_in_valid;
  logic [NUM_SAMPLER_LANES-1:0] samp_in_ready;
  logic [NUM_SAMPLER_LANES-1:0][63:0] samp_in_word;

  logic [NUM_SAMPLER_LANES-1:0] samp_out_valid;
  logic [NUM_SAMPLER_LANES-1:0] samp_out_ready;
  logic [NUM_SAMPLER_LANES-1:0][63:0] samp_out_coeff;

  // Wire issuer->sampler and backpressure
  generate
    for (i=0;i<NUM_SAMPLER_LANES;i++) begin : g_map
      localparam int unsigned shake_idx = (i / ISSUE_W_PER_SHAKE);
      localparam int unsigned slot_idx  = (i % ISSUE_W_PER_SHAKE);
      assign samp_in_valid[i] = iss_valid[shake_idx][slot_idx];
      assign samp_in_word[i]  = iss_word[shake_idx][slot_idx];
      assign iss_ready[shake_idx][slot_idx] = samp_in_ready[i];
    end
  endgenerate

  generate
    for (i=0;i<NUM_SAMPLER_LANES;i++) begin : g_samp
      logic [31:0] stat_words, stat_accepts, stat_rejects;
      reject_sampler_lane u_samp (
        .clk(clk),
        .rst_n(rst_n),
        .q(q),
        .threshold_T(threshold_T),
        .in_valid(samp_in_valid[i]),
        .in_ready(samp_in_ready[i]),
        .in_word(samp_in_word[i]),
        .out_valid(samp_out_valid[i]),
        .out_ready(samp_out_ready[i]),
        .out_coeff(samp_out_coeff[i]),
        .stat_words(stat_words),
        .stat_accepts(stat_accepts),
        .stat_rejects(stat_rejects)
      );
    end
  endgenerate

  // Output mapping
  assign coeff_valid = samp_out_valid;
  assign coeff_data  = samp_out_coeff;
  assign samp_out_ready = coeff_ready;

endmodule

