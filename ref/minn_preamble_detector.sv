// -----------------------------------------------------------------------------
// Module: minn_preamble_detector
// -----------------------------------------------------------------------------
// Two-antenna Minn-style OFDM preamble detector. Buffers incoming IQ samples,
// evaluates the running correlation and energy windows, and declares a frame
// start when the metric exceeds a programmable threshold.
// -----------------------------------------------------------------------------
module minn_preamble_detector #(
    parameter int INPUT_WIDTH       = 12,
    parameter int AXIS_DATA_WIDTH   = INPUT_WIDTH * 4,
    parameter int NFFT              = 2048,
    parameter int CP_LEN            = 512,
    parameter int OUTPUT_MARGIN     = CP_LEN,
    parameter int THRESH_VALUE      = 32'd4096,
    parameter int THRESH_FRAC_BITS  = 15,
    parameter int SMOOTH_SHIFT      = 3,
    parameter int HYSTERESIS        = 2,
    parameter int TIMING_OFFSET     = -CP_LEN,
    parameter int METRIC_DBG_WIDTH  = 24
) (
    input  logic                        clk,
    input  logic                        rst,
    // AXI4-Stream input (packed as {ch1_q, ch1_i, ch0_q, ch0_i})
    input  logic [AXIS_DATA_WIDTH-1:0] s_axis_tdata,
    input  logic                       s_axis_tvalid,
    output logic                       s_axis_tready,
    input  logic                       s_axis_tlast,
    // AXI4-Stream output (packed as {ch1_q, ch1_i, ch0_q, ch0_i})
    output logic [AXIS_DATA_WIDTH-1:0] m_axis_tdata,
    output logic                       m_axis_tvalid,
    input  logic                       m_axis_tready,
    output logic                       m_axis_tlast,
    // Frame start indicator aligned with output handshake
    output logic                       frame_start
`ifdef MINN_METRIC_DEBUG
   ,output logic [METRIC_DBG_WIDTH-1:0] metric_dbg
`endif
);
    // -------------------------------------------------------------------------
    // Derived constants and configuration checks
    // -------------------------------------------------------------------------
    localparam int QUARTER_LEN = NFFT / 4;
    initial begin
        if ((NFFT % 4) != 0) begin
            $error("NFFT must be divisible by 4 for Minn detector.");
        end
        if (OUTPUT_MARGIN < 0) begin
            $error("OUTPUT_MARGIN must be non-negative.");
        end
        if (AXIS_DATA_WIDTH != (INPUT_WIDTH * 4)) begin
            $error("AXIS_DATA_WIDTH must equal 4 * INPUT_WIDTH for dual-antenna Minn detector.");
        end
    end

    // Output buffering dimensions
    localparam int OUTPUT_DELAY = NFFT;
    localparam int OUTPUT_DEPTH = OUTPUT_DELAY + OUTPUT_MARGIN;
    localparam int OUT_ADDR_WIDTH = (OUTPUT_DEPTH <= 1) ? 1 : $clog2(OUTPUT_DEPTH);

    // Metric width calculations
    localparam int SUM_GROWTH    = (QUARTER_LEN <= 1) ? 1 : $clog2(QUARTER_LEN + 1);
    localparam int PRODUCT_WIDTH = (2 * INPUT_WIDTH) + 1;
    localparam int POWER_WIDTH   = (2 * INPUT_WIDTH) + 1;
    localparam int CORR_WIDTH    = PRODUCT_WIDTH + SUM_GROWTH;
    localparam int ENERGY_WIDTH  = POWER_WIDTH + SUM_GROWTH;
    localparam int CORR_TOTAL_WIDTH = CORR_WIDTH + 2; // Sum of four quarter correlations.
    localparam int ENERGY_TOTAL_WIDTH = ENERGY_WIDTH + 3; // Sum of six quarter energies.

    // Threshold arithmetic helpers
    localparam int HYST_WIDTH = (HYSTERESIS <= 0) ? 1 : $clog2(HYSTERESIS + 1);
    localparam logic [HYST_WIDTH-1:0] HYST_LIMIT = (HYSTERESIS <= 1)
        ? HYST_WIDTH'(0)
        : HYST_WIDTH'(HYSTERESIS - 1);
    localparam int THRESH_WIDTH = (THRESH_VALUE <= 0) ? 1 : $clog2(THRESH_VALUE + 1);
    localparam logic [THRESH_WIDTH-1:0] THRESH_CONST = THRESH_WIDTH'(THRESH_VALUE);
    localparam int THRESH_CONST_INT = THRESH_VALUE;
    localparam bit THRESH_IS_NONZERO = (THRESH_CONST_INT != 0);
    localparam bit THRESH_IS_POWER_OF_TWO = (THRESH_CONST_INT > 0)
        && ((THRESH_CONST_INT & (THRESH_CONST_INT - 1)) == 0);
    localparam int THRESH_SHIFT = THRESH_IS_POWER_OF_TWO ? $clog2(THRESH_CONST_INT) : 0;
    localparam int CORR_SCALED_WIDTH = CORR_TOTAL_WIDTH + THRESH_FRAC_BITS;
    localparam int ENERGY_SCALED_WIDTH = ENERGY_TOTAL_WIDTH + THRESH_WIDTH;
    localparam int COMP_WIDTH = (CORR_SCALED_WIDTH > ENERGY_SCALED_WIDTH)
        ? CORR_SCALED_WIDTH
        : ENERGY_SCALED_WIDTH;
    localparam int TIMING_OFFSET_MOD  = (OUTPUT_DEPTH == 0) ? 0 : (TIMING_OFFSET % OUTPUT_DEPTH);
    localparam int TIMING_OFFSET_NORM = (TIMING_OFFSET_MOD < 0)
        ? (TIMING_OFFSET_MOD + OUTPUT_DEPTH)
        : TIMING_OFFSET_MOD;
    localparam logic [OUT_ADDR_WIDTH-1:0] TIMING_OFFSET_VEC = OUT_ADDR_WIDTH'(TIMING_OFFSET_NORM);
    localparam logic [OUT_ADDR_WIDTH-1:0] WRITE_WRAP = OUT_ADDR_WIDTH'(OUTPUT_DEPTH - 1);
    localparam logic [OUT_ADDR_WIDTH-1:0] READ_WRAP  = WRITE_WRAP;
    localparam logic [OUT_ADDR_WIDTH:0]   OUTPUT_DEPTH_COUNT = (OUT_ADDR_WIDTH+1)'(OUTPUT_DEPTH);
    localparam logic [OUT_ADDR_WIDTH:0]   OUTPUT_DELAY_COUNT = (OUT_ADDR_WIDTH+1)'(OUTPUT_DELAY);

    // Sample packing layout (I/Q pairs for both antennas)
    localparam int ENTRY_WIDTH = AXIS_DATA_WIDTH;
    localparam int CH0_I_LSB    = 0;
    localparam int CH0_Q_LSB    = CH0_I_LSB + INPUT_WIDTH;
    localparam int CH1_I_LSB    = CH0_Q_LSB + INPUT_WIDTH;
    localparam int CH1_Q_LSB    = CH1_I_LSB + INPUT_WIDTH;

    (* ram_style = "block" *)
    logic [ENTRY_WIDTH-1:0] sample_mem [0:OUTPUT_DEPTH-1];
    logic                   sample_last_mem [0:OUTPUT_DEPTH-1];

    // Detection queue configuration
    localparam int DET_QUEUE_DEPTH = 4;
    localparam int DET_QUEUE_ADDR_WIDTH = (DET_QUEUE_DEPTH <= 1) ? 1 : $clog2(DET_QUEUE_DEPTH);
    localparam int DET_TIMER_WIDTH = OUT_ADDR_WIDTH + 2;
    localparam logic [DET_QUEUE_ADDR_WIDTH:0] DET_QUEUE_DEPTH_COUNT =
        (DET_QUEUE_ADDR_WIDTH+1)'(DET_QUEUE_DEPTH);

    // -------------------------------------------------------------------------
    // Sample buffering and detection queue state
    // -------------------------------------------------------------------------
    logic [DET_QUEUE_ADDR_WIDTH-1:0] detect_wr_ptr;
    logic [DET_QUEUE_ADDR_WIDTH-1:0] detect_rd_ptr;
    logic [DET_QUEUE_ADDR_WIDTH:0]   detect_count;
    logic [DET_TIMER_WIDTH-1:0]      detect_queue [0:DET_QUEUE_DEPTH-1];

    logic [OUT_ADDR_WIDTH-1:0] write_ptr;
    logic [OUT_ADDR_WIDTH-1:0] read_ptr;
    logic [OUT_ADDR_WIDTH:0]   sample_count;
    logic [OUT_ADDR_WIDTH-1:0] read_ptr_plus1;
    logic                      produce_output;
    logic [OUT_ADDR_WIDTH-1:0] read_ptr_future;
    logic [OUT_ADDR_WIDTH:0]   sample_count_incremented;
    logic [OUT_ADDR_WIDTH:0]   sample_count_future;
    logic [OUT_ADDR_WIDTH:0]   sample_gap_raw;
    logic [DET_TIMER_WIDTH-1:0] fill_gap;
    logic                      queue_has_entries;
    logic [DET_TIMER_WIDTH-1:0] detect_front;
    logic                      pop_event_req;
    logic                      dec_event_req;
    logic                      push_event_req;

    // AXI4-Stream handshake and unpacking
    logic                       sample_accept;
    logic signed [INPUT_WIDTH-1:0] in_ch0_i;
    logic signed [INPUT_WIDTH-1:0] in_ch0_q;
    logic signed [INPUT_WIDTH-1:0] in_ch1_i;
    logic signed [INPUT_WIDTH-1:0] in_ch1_q;
    logic [AXIS_DATA_WIDTH-1:0]    output_data_reg;
    logic                          output_valid_reg;
    logic                          output_last_reg;
    logic                          frame_start_pending;
    logic                          output_fire;
    logic                          load_output;

    assign sample_accept = s_axis_tvalid && s_axis_tready;
    assign in_ch0_i = $signed(s_axis_tdata[CH0_I_LSB +: INPUT_WIDTH]);
    assign in_ch0_q = $signed(s_axis_tdata[CH0_Q_LSB +: INPUT_WIDTH]);
    assign in_ch1_i = $signed(s_axis_tdata[CH1_I_LSB +: INPUT_WIDTH]);
    assign in_ch1_q = $signed(s_axis_tdata[CH1_Q_LSB +: INPUT_WIDTH]);
    assign m_axis_tdata  = output_data_reg;
    assign m_axis_tvalid = output_valid_reg;
    assign m_axis_tlast  = output_last_reg;
    assign output_fire   = output_valid_reg && m_axis_tready;

    // Circular buffer pointer helpers
    assign read_ptr_plus1 = (read_ptr == READ_WRAP) ? '0 : read_ptr + 1'b1;
    assign produce_output = sample_accept && (sample_count >= OUTPUT_DELAY_COUNT);
    assign read_ptr_future = produce_output ? read_ptr_plus1 : read_ptr;
    assign load_output = produce_output;
    assign sample_count_incremented = (sample_count < OUTPUT_DEPTH_COUNT)
        ? sample_count + 1'b1
        : sample_count;
    assign sample_count_future = sample_accept ? sample_count_incremented : sample_count;
    assign sample_gap_raw = (sample_count_future >= OUTPUT_DELAY_COUNT)
        ? '0
        : OUTPUT_DELAY_COUNT - sample_count_future;
    assign fill_gap = {{(DET_TIMER_WIDTH-(OUT_ADDR_WIDTH+1)){1'b0}}, sample_gap_raw};
    assign s_axis_tready = (!output_valid_reg) || m_axis_tready;
    assign frame_start = output_fire && frame_start_pending;
    assign queue_has_entries = (detect_count != {(DET_QUEUE_ADDR_WIDTH+1){1'b0}});
    assign detect_front = detect_queue[detect_rd_ptr];
    assign pop_event_req = produce_output && queue_has_entries && (detect_front == {DET_TIMER_WIDTH{1'b0}});
    assign dec_event_req = produce_output && queue_has_entries && (detect_front != {DET_TIMER_WIDTH{1'b0}});
    assign push_event_req = detection_pulse && (detect_count < DET_QUEUE_DEPTH_COUNT);

    // -------------------------------------------------------------------------
    // Antenna processing paths
    // -------------------------------------------------------------------------
    logic signed [CORR_WIDTH-1:0]  ch0_corr_recent;
    logic signed [CORR_WIDTH-1:0]  ch0_corr_prev;
    logic signed [ENERGY_WIDTH-1:0] ch0_energy_recent;
    logic signed [ENERGY_WIDTH-1:0] ch0_energy_prev;
    logic signed [ENERGY_WIDTH-1:0] ch0_energy_prev2;
    logic                           ch0_valid;

    logic signed [CORR_WIDTH-1:0]  ch1_corr_recent;
    logic signed [CORR_WIDTH-1:0]  ch1_corr_prev;
    logic signed [ENERGY_WIDTH-1:0] ch1_energy_recent;
    logic signed [ENERGY_WIDTH-1:0] ch1_energy_prev;
    logic signed [ENERGY_WIDTH-1:0] ch1_energy_prev2;
    logic                           ch1_valid;

    minn_antenna_path #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .QUARTER_LEN(QUARTER_LEN)
    ) antenna0 (
        .clk(clk),
        .rst(rst),
        .in_valid(sample_accept),
        .in_i(in_ch0_i),
        .in_q(in_ch0_q),
        .corr_recent(ch0_corr_recent),
        .corr_previous(ch0_corr_prev),
        .energy_recent(ch0_energy_recent),
        .energy_previous(ch0_energy_prev),
        .energy_previous2(ch0_energy_prev2),
        .taps_valid(ch0_valid)
    );

    minn_antenna_path #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .QUARTER_LEN(QUARTER_LEN)
    ) antenna1 (
        .clk(clk),
        .rst(rst),
        .in_valid(sample_accept),
        .in_i(in_ch1_i),
        .in_q(in_ch1_q),
        .corr_recent(ch1_corr_recent),
        .corr_previous(ch1_corr_prev),
        .energy_recent(ch1_energy_recent),
        .energy_previous(ch1_energy_prev),
        .energy_previous2(ch1_energy_prev2),
        .taps_valid(ch1_valid)
    );

    // Metric is valid once both antenna pipelines have produced taps.
    logic metric_valid;
    assign metric_valid = sample_accept && ch0_valid && ch1_valid;

    // -------------------------------------------------------------------------
    // Correlation and energy aggregation across antennas
    // -------------------------------------------------------------------------
    logic signed [CORR_WIDTH:0] corr_sum_ch0;
    logic signed [CORR_WIDTH:0] corr_sum_ch1;
    logic signed [CORR_TOTAL_WIDTH-1:0] corr_total;
    logic [ENERGY_WIDTH+1:0] energy_sum_ch0;
    logic [ENERGY_WIDTH+1:0] energy_sum_ch1;
    logic [ENERGY_TOTAL_WIDTH-1:0] energy_total;

    always_comb begin
        corr_sum_ch0 = $signed(ch0_corr_recent) + $signed(ch0_corr_prev);
        corr_sum_ch1 = $signed(ch1_corr_recent) + $signed(ch1_corr_prev);
        corr_total   = $signed({corr_sum_ch0[CORR_WIDTH], corr_sum_ch0})
                     + $signed({corr_sum_ch1[CORR_WIDTH], corr_sum_ch1});

        energy_sum_ch0 = {2'b00, $unsigned(ch0_energy_recent)}
                        + {2'b00, $unsigned(ch0_energy_prev)}
                        + {2'b00, $unsigned(ch0_energy_prev2)};
        energy_sum_ch1 = {2'b00, $unsigned(ch1_energy_recent)}
                        + {2'b00, $unsigned(ch1_energy_prev)}
                        + {2'b00, $unsigned(ch1_energy_prev2)};
        energy_total   = {1'b0, energy_sum_ch0} + {1'b0, energy_sum_ch1};
    end

    // -------------------------------------------------------------------------
    // Clamp correlations to positive domain
    // -------------------------------------------------------------------------
    logic [CORR_TOTAL_WIDTH-1:0] corr_positive;
    always_comb begin
        if (corr_total[CORR_TOTAL_WIDTH-1] == 1'b1) begin
            corr_positive = '0;
        end else begin
            corr_positive = corr_total[CORR_TOTAL_WIDTH-1:0];
        end
    end

    // -------------------------------------------------------------------------
    // Optional exponential smoothing of the metric
    // -------------------------------------------------------------------------
    logic [CORR_TOTAL_WIDTH-1:0] smooth_metric;
    generate
        if (SMOOTH_SHIFT == 0) begin : g_no_smooth
            always_ff @(posedge clk) begin
                if (rst) begin
                    smooth_metric <= '0;
                end else if (metric_valid) begin
                    smooth_metric <= corr_positive;
                end
            end
        end else begin : g_smooth
            logic signed [CORR_TOTAL_WIDTH:0] diff;
            logic signed [CORR_TOTAL_WIDTH:0] smooth_next;
            always_ff @(posedge clk) begin
                if (rst) begin
                    smooth_metric <= '0;
                end else if (metric_valid) begin
                    diff = $signed({1'b0, corr_positive}) - $signed({1'b0, smooth_metric});
                    smooth_next = $signed({1'b0, smooth_metric}) + (diff >>> SMOOTH_SHIFT);
                    smooth_metric <= smooth_next[CORR_TOTAL_WIDTH-1:0];
                end
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Threshold comparison via cross multiplication
    // -------------------------------------------------------------------------
    logic [CORR_SCALED_WIDTH-1:0] corr_scaled_native;
    logic [ENERGY_SCALED_WIDTH-1:0] energy_scaled_native;
    logic [ENERGY_SCALED_WIDTH-1:0] energy_total_ext;
    logic [COMP_WIDTH-1:0] corr_scaled;
    logic [COMP_WIDTH-1:0] energy_scaled;
    logic above_threshold;

    assign corr_scaled_native = {smooth_metric, {THRESH_FRAC_BITS{1'b0}}};
    assign energy_total_ext = {{(ENERGY_SCALED_WIDTH-ENERGY_TOTAL_WIDTH){1'b0}}, energy_total};
    generate
        if (!THRESH_IS_NONZERO) begin : g_zero_threshold
            assign energy_scaled_native = '0;
        end else if (THRESH_IS_POWER_OF_TWO) begin : g_power_of_two_threshold
            assign energy_scaled_native = energy_total_ext << THRESH_SHIFT;
        end else begin : g_generic_threshold
            assign energy_scaled_native = energy_total * THRESH_CONST;
        end
    endgenerate
    assign corr_scaled = {{(COMP_WIDTH-CORR_SCALED_WIDTH){1'b0}}, corr_scaled_native};
    assign energy_scaled = {{(COMP_WIDTH-ENERGY_SCALED_WIDTH){1'b0}}, energy_scaled_native};
    assign above_threshold = metric_valid && (corr_scaled >= energy_scaled);

    // -------------------------------------------------------------------------
    // Gate and peak tracking for detection
    // -------------------------------------------------------------------------
    logic gate_open;
    logic [CORR_TOTAL_WIDTH-1:0] peak_value;
    logic [OUT_ADDR_WIDTH-1:0]   peak_ptr;
    logic [HYST_WIDTH-1:0]       low_counter;
    logic                        detection_pulse;
    logic [OUT_ADDR_WIDTH-1:0]   detection_addr;

    always_ff @(posedge clk) begin
        if (rst) begin
            gate_open       <= 1'b0;
            peak_value      <= '0;
            peak_ptr        <= '0;
            low_counter     <= '0;
            detection_pulse <= 1'b0;
            detection_addr  <= '0;
        end else begin
            detection_pulse <= 1'b0;
            if (metric_valid) begin
                if (!gate_open) begin
                    low_counter <= '0;
                    if (above_threshold) begin
                        gate_open  <= 1'b1;
                        peak_value <= corr_positive;
                        peak_ptr   <= write_ptr;
                    end
                end else begin
                    // Track the highest raw (unsmoothed) metric to minimize latency bias.
                    if (corr_positive >= peak_value) begin
                        peak_value <= corr_positive;
                        peak_ptr   <= write_ptr;
                    end

                    if (above_threshold) begin
                        low_counter <= '0;
                    end else if (HYSTERESIS == 0) begin
                        gate_open       <= 1'b0;
                        peak_value      <= '0;
                        low_counter     <= '0;
                        detection_pulse <= 1'b1;
                        detection_addr  <= ptr_with_offset(peak_ptr);
                    end else begin
                        if (low_counter == HYST_LIMIT) begin
                            gate_open       <= 1'b0;
                            peak_value      <= '0;
                            low_counter     <= '0;
                            detection_pulse <= 1'b1;
                            detection_addr  <= ptr_with_offset(peak_ptr);
                        end else begin
                            low_counter <= low_counter + 1'b1;
                        end
                    end
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Pointer arithmetic helpers
    // -------------------------------------------------------------------------
    function automatic logic [OUT_ADDR_WIDTH-1:0] ptr_with_offset(
        input logic [OUT_ADDR_WIDTH-1:0] base
    );
        logic [OUT_ADDR_WIDTH:0] sum_vec;
        begin
            sum_vec = {1'b0, base} + {1'b0, TIMING_OFFSET_VEC};
            if (sum_vec >= OUTPUT_DEPTH_COUNT) begin
                sum_vec = sum_vec - OUTPUT_DEPTH_COUNT;
            end
            ptr_with_offset = sum_vec[OUT_ADDR_WIDTH-1:0];
        end
    endfunction

    function automatic logic [DET_TIMER_WIDTH-1:0] ring_distance(
        input logic [OUT_ADDR_WIDTH-1:0] target,
        input logic [OUT_ADDR_WIDTH-1:0] origin
    );
        logic [DET_TIMER_WIDTH-1:0] target_ext;
        logic [DET_TIMER_WIDTH-1:0] origin_ext;
        begin
            target_ext = DET_TIMER_WIDTH'({1'b0, target});
            origin_ext = DET_TIMER_WIDTH'({1'b0, origin});
            if (target >= origin) begin
                ring_distance = target_ext - origin_ext;
            end else begin
                ring_distance = target_ext + DET_TIMER_WIDTH'(OUTPUT_DEPTH) - origin_ext;
            end
        end
    endfunction

`ifdef MINN_METRIC_DEBUG
    // -------------------------------------------------------------------------
    // Metric debug truncation helper
    // -------------------------------------------------------------------------
    function automatic logic [METRIC_DBG_WIDTH-1:0] metric_trunc(
        input logic [CORR_TOTAL_WIDTH-1:0] value
    );
        logic [METRIC_DBG_WIDTH-1:0] result;
        int start_index;
        start_index = (CORR_TOTAL_WIDTH > METRIC_DBG_WIDTH)
            ? (CORR_TOTAL_WIDTH - METRIC_DBG_WIDTH)
            : 0;
        result = '0;
        for (int idx = 0; idx < METRIC_DBG_WIDTH; idx++) begin
            int source = start_index + idx;
            if (source < CORR_TOTAL_WIDTH) begin
                result[idx] = value[source];
            end
        end
        return result;
    endfunction
`endif

    // -------------------------------------------------------------------------
    // Output buffer management
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            write_ptr           <= '0;
            read_ptr            <= '0;
            sample_count        <= '0;
            output_data_reg     <= '0;
            output_last_reg     <= 1'b0;
            output_valid_reg    <= 1'b0;
            frame_start_pending <= 1'b0;
            detect_wr_ptr       <= '0;
            detect_rd_ptr       <= '0;
            detect_count        <= '0;
            for (int q = 0; q < DET_QUEUE_DEPTH; q++) begin
                detect_queue[q] <= '0;
            end
`ifdef MINN_METRIC_DEBUG
            metric_dbg          <= '0;
`endif
        end else begin
            if (output_fire) begin
                output_valid_reg <= 1'b0;
            end
            if (load_output) begin
                output_valid_reg    <= 1'b1;
                output_data_reg     <= sample_mem[read_ptr];
                output_last_reg     <= sample_last_mem[read_ptr];
                read_ptr            <= read_ptr_plus1;
            end else if (output_fire) begin
                output_last_reg <= 1'b0;
            end

            if (load_output) begin
                frame_start_pending <= pop_event_req;
            end else if (output_fire) begin
                frame_start_pending <= 1'b0;
            end

            if (sample_accept) begin
                sample_mem[write_ptr]      <= s_axis_tdata;
                sample_last_mem[write_ptr] <= s_axis_tlast;
                if (write_ptr == WRITE_WRAP) begin
                    write_ptr <= '0;
                end else begin
                    write_ptr <= write_ptr + 1'b1;
                end
                if (sample_count < OUTPUT_DEPTH_COUNT) begin
                    sample_count <= sample_count + 1'b1;
                end
            end

            if (dec_event_req) begin
                detect_queue[detect_rd_ptr] <= detect_front - 1'b1;
            end

            if (pop_event_req) begin
                if (detect_rd_ptr == DET_QUEUE_ADDR_WIDTH'(DET_QUEUE_DEPTH-1)) begin
                    detect_rd_ptr <= '0;
                end else begin
                    detect_rd_ptr <= detect_rd_ptr + 1'b1;
                end
            end

            if (push_event_req) begin
                detect_queue[detect_wr_ptr] <= ring_distance(detection_addr, read_ptr_future) + fill_gap;
                if (detect_wr_ptr == DET_QUEUE_ADDR_WIDTH'(DET_QUEUE_DEPTH-1)) begin
                    detect_wr_ptr <= '0;
                end else begin
                    detect_wr_ptr <= detect_wr_ptr + 1'b1;
                end
            end

            detect_count <= detect_count
                + (push_event_req ? {{DET_QUEUE_ADDR_WIDTH{1'b0}}, 1'b1} : '0)
                - (pop_event_req  ? {{DET_QUEUE_ADDR_WIDTH{1'b0}}, 1'b1} : '0);

`ifdef MINN_METRIC_DEBUG
            if (metric_valid) begin
                if (SMOOTH_SHIFT == 0) begin
                    metric_dbg <= metric_trunc(corr_positive);
                end else begin
                    metric_dbg <= metric_trunc(smooth_metric);
                end
            end
`endif
        end
    end
endmodule

`default_nettype wire
