// -----------------------------------------------------------------------------
// Module: minn_delay_line
// -----------------------------------------------------------------------------
// Streaming delay line with a parameterized width and depth. Samples are written
// when `in_valid` is asserted and emerge after DEPTH cycles once the pipeline
// fills. Output is held invalid while the memory priming phase completes.
// -----------------------------------------------------------------------------
module minn_delay_line #(
    parameter int WIDTH = 16,
    parameter int DEPTH = 1
) (
    input  logic                      clk,
    input  logic                      rst,
    input  logic                      in_valid,
    input  logic signed [WIDTH-1:0]   in_data,
    output logic                      out_valid,
    output logic signed [WIDTH-1:0]   out_data
);

    // -------------------------------------------------------------------------
    // Parameter validation
    // -------------------------------------------------------------------------
    if (DEPTH <= 0) begin : g_invalid_depth
        initial begin
            $error("minn_delay_line DEPTH must be positive.");
        end
    end else begin : g_delay
        // ---------------------------------------------------------------------
        // Derived types and state encoding
        // ---------------------------------------------------------------------
        localparam int ADDR_WIDTH  = (DEPTH <= 1) ? 1 : $clog2(DEPTH);
        localparam int COUNT_WIDTH = (DEPTH <= 1) ? 1 : $clog2(DEPTH + 1);

        typedef logic [ADDR_WIDTH-1:0]  addr_t;
        typedef logic [COUNT_WIDTH-1:0] count_t;

        localparam addr_t  LAST_ADDR   = addr_t'(DEPTH - 1);
        localparam count_t DEPTH_COUNT = count_t'(DEPTH);

        (* ram_style = "block" *)
        logic signed [WIDTH-1:0] mem [0:DEPTH-1];

        addr_t  wr_ptr;
        count_t fill_count;

        // ---------------------------------------------------------------------
        // Write / read pipeline
        // ---------------------------------------------------------------------
        always_ff @(posedge clk) begin
            if (rst) begin
                wr_ptr     <= '0;
                fill_count <= '0;
                out_data   <= '0;
                out_valid  <= 1'b0;
            end else if (in_valid) begin
                logic signed [WIDTH-1:0] read_data;

                read_data = (fill_count < DEPTH_COUNT) ? '0 : mem[wr_ptr];
                mem[wr_ptr] <= in_data;

                if (wr_ptr == LAST_ADDR) begin
                    wr_ptr <= '0;
                end else begin
                    wr_ptr <= wr_ptr + 1'b1;
                end

                if (fill_count < DEPTH_COUNT) begin
                    fill_count <= fill_count + count_t'(1);
                    out_valid  <= 1'b0;
                    out_data   <= '0;
                end else begin
                    out_valid <= 1'b1;
                    out_data  <= read_data;
                end
            end else begin
                out_valid <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
