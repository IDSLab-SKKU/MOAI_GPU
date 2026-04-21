// Descriptor pack / domain separation
// Packs fields into per-lane fixed-width descriptor bits.
// Bit ordering (LSB-first in the packed vector):
//   [ 255:0]   master_seed
//   [ 319:256] key_id
//   [ 383:320] decomp_id
//   [ 447:384] limb_id
//   [ 511:448] lane_id
//   [ 639:512] domain_tag (128b)
module prng_descriptor_pack #(
  parameter int unsigned SEED_W = 256,
  parameter int unsigned KEY_ID_W = 64,
  parameter int unsigned DECOMP_ID_W = 64,
  parameter int unsigned LIMB_ID_W = 64,
  parameter int unsigned LANE_ID_W = 64,
  parameter int unsigned DOMAIN_TAG_W = 128,
  parameter int unsigned DESC_W = (SEED_W+KEY_ID_W+DECOMP_ID_W+LIMB_ID_W+LANE_ID_W+DOMAIN_TAG_W),
  parameter int unsigned NUM_LANES = 2
) (
  input  logic [DOMAIN_TAG_W-1:0] domain_tag,
  input  logic [SEED_W-1:0]       master_seed,
  input  logic [KEY_ID_W-1:0]     key_id,
  input  logic [DECOMP_ID_W-1:0]  decomp_id,
  input  logic [LIMB_ID_W-1:0]    limb_id,
  input  logic [LANE_ID_W-1:0]    lane_id_base,
  output logic [NUM_LANES-1:0][DESC_W-1:0] lane_desc
);
  // Lane id is derived as lane_id_base + lane_index (simple deterministic scheme)
  genvar i;
  generate
    for (i=0;i<NUM_LANES;i++) begin : g
      logic [LANE_ID_W-1:0] lane_id;
      assign lane_id = lane_id_base + LANE_ID_W'(i);

      // Pack LSB-first for readability of slicing.
      assign lane_desc[i] = {
        domain_tag,
        lane_id,
        limb_id,
        decomp_id,
        key_id,
        master_seed
      };
    end
  endgenerate
endmodule

