// -----------------------------------------------------------------------------
// Module: minn_antenna_path
// -----------------------------------------------------------------------------
// Implements the Minn correlator primitives for a single antenna stream. The
// block generates quarter-delayed samples, correlation taps, and energy metrics
// that feed the preamble detector.
// -----------------------------------------------------------------------------
module minn_antenna_path #(
    parameter int INPUT_WIDTH   = 12,
    parameter int QUARTER_LEN   = 512,
    // Derived parameters - do not override at instantiation.
    parameter int SUM_GROWTH    = (QUARTER_LEN <= 1) ? 1 : $clog2(QUARTER_LEN + 1),
    parameter int PRODUCT_WIDTH = (2 * INPUT_WIDTH) + 1,
    parameter int POWER_WIDTH   = (2 * INPUT_WIDTH) + 1,
    parameter int CORR_WIDTH    = PRODUCT_WIDTH + SUM_GROWTH,
    parameter int ENERGY_WIDTH  = POWER_WIDTH + SUM_GROWTH
) (
    input  logic                          clk,
    input  logic                          rst,
    input  logic                          in_valid,
    input  logic signed [INPUT_WIDTH-1:0] in_i,
    input  logic signed [INPUT_WIDTH-1:0] in_q,
    output logic signed [CORR_WIDTH-1:0]  corr_recent,
    output logic signed [CORR_WIDTH-1:0]  corr_previous,
    output logic signed [ENERGY_WIDTH-1:0] energy_recent,
    output logic signed [ENERGY_WIDTH-1:0] energy_previous,
    output logic signed [ENERGY_WIDTH-1:0] energy_previous2,
    output logic                          taps_valid
);
    // -------------------------------------------------------------------------
    // Quarter-delayed input taps
    // -------------------------------------------------------------------------
    logic signed [INPUT_WIDTH-1:0] delayed_i;
    logic signed [INPUT_WIDTH-1:0] delayed_q;

    minn_delay_line #(
        .WIDTH(INPUT_WIDTH),
        .DEPTH(QUARTER_LEN)
    ) delay_i (
        .clk(clk),
        .rst(rst),
        .in_valid(in_valid),
        .in_data(in_i),
        .out_valid(),
        .out_data(delayed_i)
    );

    minn_delay_line #(
        .WIDTH(INPUT_WIDTH),
        .DEPTH(QUARTER_LEN)
    ) delay_q (
        .clk(clk),
        .rst(rst),
        .in_valid(in_valid),
        .in_data(in_q),
        .out_valid(),
        .out_data(delayed_q)
    );

    // -------------------------------------------------------------------------
    // Quarter-lag correlation product
    // -------------------------------------------------------------------------
    logic signed [2*INPUT_WIDTH-1:0] prod_ii;
    logic signed [2*INPUT_WIDTH-1:0] prod_qq;
    logic signed [PRODUCT_WIDTH-1:0] quarter_product;

    assign prod_ii = delayed_i * in_i;
    assign prod_qq = delayed_q * in_q;
    assign quarter_product = $signed({prod_ii[2*INPUT_WIDTH-1], prod_ii})
                           + $signed({prod_qq[2*INPUT_WIDTH-1], prod_qq});

    // -------------------------------------------------------------------------
    // Instantaneous energy accumulation
    // -------------------------------------------------------------------------
    logic signed [2*INPUT_WIDTH-1:0] power_i;
    logic signed [2*INPUT_WIDTH-1:0] power_q;
    logic signed [POWER_WIDTH-1:0]   power_sum;

    assign power_i = in_i * in_i;
    assign power_q = in_q * in_q;
    assign power_sum = $signed({1'b0, power_i}) + $signed({1'b0, power_q});

    // -------------------------------------------------------------------------
    // Running sum of quarter products
    // -------------------------------------------------------------------------
    logic signed [CORR_WIDTH-1:0] corr_sum;
    logic                         corr_valid;

    minn_running_sum #(
        .WIDTH(PRODUCT_WIDTH),
        .DEPTH(QUARTER_LEN)
    ) corr_window (
        .clk(clk),
        .rst(rst),
        .in_valid(in_valid),
        .sample_in(quarter_product),
        .sum_out(corr_sum),
        .sum_valid(corr_valid)
    );

    // -------------------------------------------------------------------------
    // Running sum of instantaneous power
    // -------------------------------------------------------------------------
    logic signed [ENERGY_WIDTH-1:0] energy_sum;
    logic                           energy_valid;

    minn_running_sum #(
        .WIDTH(POWER_WIDTH),
        .DEPTH(QUARTER_LEN)
    ) energy_window (
        .clk(clk),
        .rst(rst),
        .in_valid(in_valid),
        .sample_in(power_sum),
        .sum_out(energy_sum),
        .sum_valid(energy_valid)
    );

    // -------------------------------------------------------------------------
    // Historical taps for metric alignment
    // -------------------------------------------------------------------------
    logic signed [CORR_WIDTH-1:0]  corr_delay_data;
    logic                           corr_delay_valid;
    logic signed [ENERGY_WIDTH-1:0] energy_delay_q;
    logic                           energy_delay_q_valid;
    logic signed [ENERGY_WIDTH-1:0] energy_delay_2q;
    logic                           energy_delay_2q_valid;

    minn_delay_line #(
        .WIDTH(CORR_WIDTH),
        .DEPTH(QUARTER_LEN)
    ) corr_delay (
        .clk(clk),
        .rst(rst),
        .in_valid(corr_valid),
        .in_data(corr_sum),
        .out_valid(corr_delay_valid),
        .out_data(corr_delay_data)
    );

    minn_delay_line #(
        .WIDTH(ENERGY_WIDTH),
        .DEPTH(QUARTER_LEN)
    ) energy_delay1 (
        .clk(clk),
        .rst(rst),
        .in_valid(energy_valid),
        .in_data(energy_sum),
        .out_valid(energy_delay_q_valid),
        .out_data(energy_delay_q)
    );

    minn_delay_line #(
        .WIDTH(ENERGY_WIDTH),
        .DEPTH(QUARTER_LEN)
    ) energy_delay2 (
        .clk(clk),
        .rst(rst),
        .in_valid(energy_delay_q_valid),
        .in_data(energy_delay_q),
        .out_valid(energy_delay_2q_valid),
        .out_data(energy_delay_2q)
    );

    // -------------------------------------------------------------------------
    // Output register update
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            corr_recent      <= '0;
            corr_previous    <= '0;
            energy_recent    <= '0;
            energy_previous  <= '0;
            energy_previous2 <= '0;
            taps_valid       <= 1'b0;
        end else begin
            if (corr_valid) begin
                corr_recent <= corr_sum;
            end
            if (corr_delay_valid) begin
                corr_previous <= corr_delay_data;
            end
            if (energy_valid) begin
                energy_recent <= energy_sum;
            end
            if (energy_delay_q_valid) begin
                energy_previous <= energy_delay_q;
            end
            if (energy_delay_2q_valid) begin
                energy_previous2 <= energy_delay_2q;
            end
            taps_valid <= energy_delay_2q_valid;
        end
    end
endmodule

`default_nettype wire
