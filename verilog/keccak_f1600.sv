// Minimal Keccak-f[1600] permutation (sequential 24 rounds).
// Not optimized; intended for architecture/sample RTL.
module keccak_f1600 (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         start,
  input  logic [1599:0] state_in,
  output logic         done,
  output logic [1599:0] state_out
);
  // State as 5x5 lanes of 64b, packed into 1600b.
  logic [63:0] A [0:4][0:4];
  logic [63:0] B [0:4][0:4];
  logic [63:0] C [0:4];
  logic [63:0] D [0:4];

  function automatic logic [63:0] rol64(input logic [63:0] x, input int unsigned n);
    rol64 = (x << n) | (x >> (64-n));
  endfunction

  // Rotation offsets r[x][y]
  function automatic int unsigned rho(input int unsigned x, input int unsigned y);
    case ({x[2:0],y[2:0]})
      6'o00: rho = 0;
      6'o10: rho = 1;
      6'o20: rho = 62;
      6'o30: rho = 28;
      6'o40: rho = 27;
      6'o01: rho = 36;
      6'o11: rho = 44;
      6'o21: rho = 6;
      6'o31: rho = 55;
      6'o41: rho = 20;
      6'o02: rho = 3;
      6'o12: rho = 10;
      6'o22: rho = 43;
      6'o32: rho = 25;
      6'o42: rho = 39;
      6'o03: rho = 41;
      6'o13: rho = 45;
      6'o23: rho = 15;
      6'o33: rho = 21;
      6'o43: rho = 8;
      6'o04: rho = 18;
      6'o14: rho = 2;
      6'o24: rho = 61;
      6'o34: rho = 56;
      6'o44: rho = 14;
      default: rho = 0;
    endcase
  endfunction

  // Round constants
  logic [63:0] RC [0:23];
  initial begin
    RC[ 0]=64'h0000000000000001; RC[ 1]=64'h0000000000008082;
    RC[ 2]=64'h800000000000808a; RC[ 3]=64'h8000000080008000;
    RC[ 4]=64'h000000000000808b; RC[ 5]=64'h0000000080000001;
    RC[ 6]=64'h8000000080008081; RC[ 7]=64'h8000000000008009;
    RC[ 8]=64'h000000000000008a; RC[ 9]=64'h0000000000000088;
    RC[10]=64'h0000000080008009; RC[11]=64'h000000008000000a;
    RC[12]=64'h000000008000808b; RC[13]=64'h800000000000008b;
    RC[14]=64'h8000000000008089; RC[15]=64'h8000000000008003;
    RC[16]=64'h8000000000008002; RC[17]=64'h8000000000000080;
    RC[18]=64'h000000000000800a; RC[19]=64'h800000008000000a;
    RC[20]=64'h8000000080008081; RC[21]=64'h8000000000008080;
    RC[22]=64'h0000000080000001; RC[23]=64'h8000000080008008;
  end

  typedef enum logic [1:0] {IDLE, RUN, OUT} st_t;
  st_t st;
  logic [4:0] round;

  // Unpack and pack helpers (lane order: x + 5*y)
  task automatic unpack_state(input logic [1599:0] s);
    int x,y;
    for (y=0;y<5;y++) begin
      for (x=0;x<5;x++) begin
        A[x][y] = s[64*(x+5*y) +: 64];
      end
    end
  endtask

  function automatic logic [1599:0] pack_state;
    logic [1599:0] s;
    int x,y;
    begin
      s = '0;
      for (y=0;y<5;y++) begin
        for (x=0;x<5;x++) begin
          s[64*(x+5*y) +: 64] = A[x][y];
        end
      end
      pack_state = s;
    end
  endfunction

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= IDLE;
      round <= 0;
      done <= 1'b0;
      state_out <= '0;
    end else begin
      done <= 1'b0;
      case (st)
        IDLE: begin
          if (start) begin
            unpack_state(state_in);
            round <= 0;
            st <= RUN;
          end
        end
        RUN: begin
          // Theta
          for (int x=0;x<5;x++) begin
            C[x] = A[x][0]^A[x][1]^A[x][2]^A[x][3]^A[x][4];
          end
          for (int x=0;x<5;x++) begin
            D[x] = C[(x+4)%5] ^ rol64(C[(x+1)%5], 1);
          end
          for (int x=0;x<5;x++) begin
            for (int y=0;y<5;y++) begin
              A[x][y] <= A[x][y] ^ D[x];
            end
          end

          // Rho+Pi into B
          for (int x=0;x<5;x++) begin
            for (int y=0;y<5;y++) begin
              int nx = y;
              int ny = (2*x + 3*y) % 5;
              B[nx][ny] = rol64(A[x][y], rho(x,y));
            end
          end

          // Chi back to A
          for (int y=0;y<5;y++) begin
            for (int x=0;x<5;x++) begin
              A[x][y] <= B[x][y] ^ ((~B[(x+1)%5][y]) & B[(x+2)%5][y]);
            end
          end

          // Iota
          A[0][0] <= A[0][0] ^ RC[round];

          if (round == 23) begin
            st <= OUT;
          end else begin
            round <= round + 1;
          end
        end
        OUT: begin
          state_out <= pack_state();
          done <= 1'b1;
          st <= IDLE;
        end
        default: st <= IDLE;
      endcase
    end
  end

endmodule

