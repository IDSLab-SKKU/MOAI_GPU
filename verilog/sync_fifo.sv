module sync_fifo #(
  parameter int unsigned W = 64,
  parameter int unsigned DEPTH = 256
) (
  input  logic clk,
  input  logic rst_n,

  input  logic wr_en,
  input  logic [W-1:0] wr_data,
  output logic full,

  input  logic rd_en,
  output logic [W-1:0] rd_data,
  output logic empty,

  output logic [$clog2(DEPTH+1)-1:0] level
);
  localparam int unsigned AW = $clog2(DEPTH);

  logic [W-1:0] mem [0:DEPTH-1];
  logic [AW-1:0] wptr, rptr;
  logic [$clog2(DEPTH+1)-1:0] count;

  assign empty = (count == 0);
  assign full  = (count == DEPTH);
  assign level = count;
  assign rd_data = mem[rptr];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wptr <= '0;
      rptr <= '0;
      count <= '0;
    end else begin
      // write
      if (wr_en && !full) begin
        mem[wptr] <= wr_data;
        wptr <= (wptr == DEPTH-1) ? '0 : (wptr + 1);
      end
      // read
      if (rd_en && !empty) begin
        rptr <= (rptr == DEPTH-1) ? '0 : (rptr + 1);
      end
      // count update
      unique case ({(wr_en && !full), (rd_en && !empty)})
        2'b10: count <= count + 1;
        2'b01: count <= count - 1;
        default: count <= count;
      endcase
    end
  end

endmodule

