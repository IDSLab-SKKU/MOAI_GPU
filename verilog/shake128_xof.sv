// SHAKE128 XOF lane (minimal, non-optimized).
// - Absorb a fixed-size seed blob (up to 512 bits in this sample interface), then finalize with SHAKE domain sep.
// - Squeeze 64-bit words (rate=168 bytes).
//
// Notes:
// - Real design would support streaming absorb; here we keep it simple for sample RTL.
// - This module uses keccak_f1600 sequential permutation.
module shake128_xof #(
  parameter int unsigned SEED_W = 512
) (
  input  logic             clk,
  input  logic             rst_n,

  input  logic             seed_valid,
  output logic             seed_ready,
  input  logic [SEED_W-1:0] seed_bits,
  input  logic [15:0]      seed_bytes,   // number of bytes in seed_bits to absorb

  input  logic             squeeze_en,
  output logic             word_valid,
  input  logic             word_ready,
  output logic [63:0]      word_data,

  output logic             busy
);
  // Keccak state (25 lanes of 64b = 1600b)
  logic [1599:0] state, state_next;
  logic perm_start, perm_done;
  logic [1599:0] perm_out;

  keccak_f1600 u_perm(
    .clk(clk), .rst_n(rst_n),
    .start(perm_start),
    .state_in(state),
    .done(perm_done),
    .state_out(perm_out)
  );

  localparam int unsigned RATE_BYTES = 168;
  localparam int unsigned RATE_WORDS = RATE_BYTES/8; // 21

  typedef enum logic [2:0] {IDLE, ABSORB, PERM0, FINALIZE, SQUEEZE, PERM_SQ} st_t;
  st_t st;

  // Squeeze cursor within rate portion
  logic [4:0] word_idx;

  function automatic logic [63:0] get_lane(input logic [1599:0] s, input int unsigned idx);
    get_lane = s[64*idx +: 64];
  endfunction

  // XOR a single byte into the state at rate offset byte_off.
  function automatic logic [1599:0] xor_byte(
    input logic [1599:0] s,
    input int unsigned byte_off,
    input logic [7:0] b
  );
    logic [1599:0] t;
    int unsigned lane;
    int unsigned off;
    begin
      t = s;
      lane = byte_off / 8;
      off  = byte_off % 8;
      t[64*lane + 8*off +: 8] = t[64*lane + 8*off +: 8] ^ b;
      xor_byte = t;
    end
  endfunction

  // Pad and permute for SHAKE: 0x1F at current pos, and 0x80 at last rate byte.
  function automatic logic [1599:0] apply_shake_pad(input logic [1599:0] s, input int unsigned pos);
    logic [1599:0] t;
    begin
      t = s;
      t = xor_byte(t, pos, 8'h1F);
      t = xor_byte(t, RATE_BYTES-1, 8'h80);
      apply_shake_pad = t;
    end
  endfunction

  // Very simple absorb: absorb seed_bytes from seed_bits little-endian byte order.
  logic [15:0] absorb_pos;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= IDLE;
      state <= '0;
      perm_start <= 1'b0;
      word_valid <= 1'b0;
      word_data <= '0;
      word_idx <= '0;
      absorb_pos <= '0;
    end else begin
      perm_start <= 1'b0;
      word_valid <= 1'b0;

      case (st)
        IDLE: begin
          state <= '0;
          absorb_pos <= 0;
          word_idx <= 0;
          if (seed_valid) begin
            st <= ABSORB;
          end
        end

        ABSORB: begin
          // absorb up to one byte per cycle (sample RTL)
          if (absorb_pos < seed_bytes) begin
            logic [7:0] b;
            b = seed_bits[8*absorb_pos +: 8];
            state <= xor_byte(state, absorb_pos, b);
            absorb_pos <= absorb_pos + 1;
          end else begin
            // apply pad, then permute once
            state <= apply_shake_pad(state, absorb_pos);
            st <= PERM0;
          end
        end

        PERM0: begin
          perm_start <= 1'b1;
          st <= FINALIZE;
        end

        FINALIZE: begin
          if (perm_done) begin
            state <= perm_out;
            word_idx <= 0;
            st <= SQUEEZE;
          end
        end

        SQUEEZE: begin
          if (squeeze_en) begin
            if (word_ready) begin
              word_valid <= 1'b1;
              word_data <= get_lane(state, word_idx);
              if (word_idx == RATE_WORDS-1) begin
                st <= PERM_SQ;
              end else begin
                word_idx <= word_idx + 1;
              end
            end else begin
              // Hold: present valid only when ready in this simple model
              word_valid <= 1'b0;
            end
          end
        end

        PERM_SQ: begin
          perm_start <= 1'b1;
          st <= (perm_done ? SQUEEZE : PERM_SQ);
          if (perm_done) begin
            state <= perm_out;
            word_idx <= 0;
            st <= SQUEEZE;
          end
        end

        default: st <= IDLE;
      endcase
    end
  end

  assign seed_ready = (st == IDLE);
  assign busy = (st != IDLE);

endmodule

