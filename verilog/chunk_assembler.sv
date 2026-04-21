module chunk_assembler #(
  parameter int unsigned COEFF_W = 64,
  parameter int unsigned CHUNK_SIZE = 32
) (
  input  logic clk,
  input  logic rst_n,

  // coeff input stream
  input  logic             s_valid,
  output logic             s_ready,
  input  logic [COEFF_W-1:0] s_coeff,
  input  logic             s_last, // asserted on last coeff of request

  // AXI-stream chunk out
  output logic             m_valid,
  input  logic             m_ready,
  output logic [CHUNK_SIZE*COEFF_W-1:0] m_data,
  output logic             m_last
);
  logic [$clog2(CHUNK_SIZE+1)-1:0] fill;
  logic [CHUNK_SIZE*COEFF_W-1:0] buf;
  logic last_seen;

  assign s_ready = (!m_valid) && (fill < CHUNK_SIZE);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      fill <= '0;
      buf <= '0;
      m_valid <= 1'b0;
      m_data <= '0;
      m_last <= 1'b0;
      last_seen <= 1'b0;
    end else begin
      // output handshake
      if (m_valid && m_ready) begin
        m_valid <= 1'b0;
        m_last <= 1'b0;
      end

      // consume coeffs
      if (s_valid && s_ready) begin
        buf[COEFF_W*fill +: COEFF_W] <= s_coeff;
        fill <= fill + 1;
        if (s_last) last_seen <= 1'b1;
      end

      // emit chunk when full, or when last coeff arrived (partial chunk)
      if (!m_valid) begin
        if (fill == CHUNK_SIZE) begin
          m_valid <= 1'b1;
          m_data <= buf;
          m_last <= last_seen;
          fill <= '0;
          buf <= '0;
          last_seen <= 1'b0;
        end else if (last_seen && (fill != 0)) begin
          m_valid <= 1'b1;
          m_data <= buf;
          m_last <= 1'b1;
          fill <= '0;
          buf <= '0;
          last_seen <= 1'b0;
        end
      end
    end
  end

endmodule

