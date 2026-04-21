package otf_keygen_pkg;
  // Coeff width (deliver u64 coeffs; downstream can truncate)
  parameter int unsigned COEFF_W = 64;
  parameter int unsigned CHUNK_SIZE = 32;

  // Lane configuration (default: 2 SHAKE lanes -> 8 sampler lanes)
  // Must satisfy: NUM_SAMPLER_LANES % NUM_SHAKE_LANES == 0
  parameter int unsigned NUM_SHAKE_LANES = 2;
  parameter int unsigned NUM_SAMPLER_LANES = 8;

  // Derived: how many 64-bit candidate words per SHAKE lane per cycle
  // (for static mapping into sampler lanes).
  parameter int unsigned ISSUE_W_PER_SHAKE = (NUM_SAMPLER_LANES / NUM_SHAKE_LANES);

  parameter int unsigned BITFIFO_DEPTH_WORDS = 256; // per SHAKE lane (64-bit words)
  parameter int unsigned COEFF_FIFO_DEPTH   = 1024; 

  typedef struct packed {
    logic [255:0] master_seed;
    logic [63:0]  key_id;
    logic [63:0]  decomp_id;
    logic [63:0]  limb_id;
    logic [63:0]  poly_id;
    logic [63:0]  q;
    logic [63:0]  threshold_T; // optional: floor(2^64/q)*q; if 0, compute in sampler (slow)
    logic [63:0]  num_coeffs;
  } prng_req_t;

  // AXI-stream chunk payload: CHUNK_SIZE * COEFF_W bits
  typedef struct packed {
    logic [CHUNK_SIZE*COEFF_W-1:0] data;
    logic                          last;
  } axis_chunk_t;

endpackage

