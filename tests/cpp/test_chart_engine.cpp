#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "../../src/ai/onnx_engine.h"
#include "../../src/risk/order_state_machine.h"
#include "../../src/ui/chart_feed.h"
#include "../../src/ui/chart_series.h"
#include "../../src/ui/confidence_corridor.h"
#include "../../src/ui/ghost_candle_overlay.h"
#include "../../src/ui/order_canvas_overlay.h"

void test_chart_series_and_viewport() {
    using namespace clow::ui;

    ChartSeries series(500);
    assert(series.empty());

    // Populate series with 10 bars
    for (int i = 0; i < 10; ++i) {
        CandleBar bar;
        bar.timestamp_ms = 1000 * i * 60;
        bar.open = 1.0800 + i * 0.0010;
        bar.high = bar.open + 0.0015;
        bar.low = bar.open - 0.0010;
        bar.close = bar.open + 0.0005;
        bar.volume = 100.0;
        series.push_bar(bar);
    }

    assert(series.size() == 10);
    assert(series.get_bar(0).open == 1.0800);
    assert(series.get_bar(9).close == 1.0895);

    double min_p = 0.0, max_p = 0.0;
    series.compute_viewport_bounds(0, 10, min_p, max_p, 0.0);
    assert(std::abs(min_p - (1.0800 - 0.0010)) < 1e-5);
    assert(std::abs(max_p - (1.0890 + 0.0015)) < 1e-5);

    // Test ViewportTransform
    ViewportTransform vt;
    vt.min_price = 1.0000;
    vt.max_price = 2.0000;
    vt.canvas_height_px = 500;
    vt.canvas_width_px = 1000;

    // Price 1.5000 is halfway -> y = 250 px
    assert(std::abs(vt.price_to_y(1.5000) - 250.0) < 1e-4);
    assert(std::abs(vt.y_to_price(250.0) - 1.5000) < 1e-4);

    std::cout << "[PASS] test_chart_series_and_viewport" << std::endl;
}

void test_chart_feed_tick_aggregation() {
    using namespace clow::ui;

    ChartSeries series;
    ChartFeed feed(TimeframePeriod::M1); // 60,000 ms

    int64_t t0 = 1700000000000; // Aligned base timestamp
    t0 = (t0 / 60000) * 60000;

    // Tick 1 in Bar 0
    feed.on_tick("EURUSD", 1.0800, 1.0802, t0 + 1000, series);
    assert(series.size() == 1);
    assert(series.get_bar(0).open == 1.0801);

    // Tick 2 in Bar 0 (higher price)
    feed.on_tick("EURUSD", 1.0820, 1.0822, t0 + 15000, series);
    assert(series.size() == 1);
    assert(series.get_bar(0).high == 1.0821);
    assert(series.get_bar(0).close == 1.0821);

    // Tick 3 in Bar 1 (new bar after 60s)
    bool new_bar = feed.on_tick("EURUSD", 1.0815, 1.0817, t0 + 61000, series);
    assert(new_bar);
    assert(series.size() == 2);
    assert(series.get_bar(1).open == 1.0816);

    std::cout << "[PASS] test_chart_feed_tick_aggregation" << std::endl;
}

void test_ghost_candle_overlay() {
    using namespace clow::ui;
    using namespace clow::ai;

    GhostCandleOverlay overlay;
    assert(!overlay.is_active());

    ForecasterPrediction pred;
    pred.valid = true;
    pred.direction_prob = 0.82f; // Bullish
    pred.anatomy = {0.70f, 0.15f, 0.15f, 1.50f};
    pred.quantiles_high = {0.5f, 1.8f, 3.0f};
    pred.quantiles_low = {0.3f, 0.8f, 1.5f};

    overlay.update_forecast(1.0850, 0.0020, pred);
    assert(overlay.is_active());
    assert(overlay.ghost_candle().is_bullish);
    assert(overlay.ghost_candle().open == 1.0850);
    assert(overlay.ghost_candle().high > 1.0850);
    assert(overlay.ghost_candle().low < 1.0850);
    assert(overlay.ghost_candle().close > 1.0850);

    ViewportTransform vt;
    vt.min_price = 1.0800;
    vt.max_price = 1.0900;
    vt.canvas_height_px = 600;
    auto rect = overlay.compute_render_geometry(vt, 10);

    assert(rect.body_width > 0.0);
    assert(rect.body_height > 0.0);
    assert(rect.wick_top_y < rect.wick_bottom_y); // Screen coords: top is smaller y

    std::cout << "[PASS] test_ghost_candle_overlay" << std::endl;
}

void test_confidence_corridor() {
    using namespace clow::ui;

    ConfidenceCorridor corridor;
    assert(!corridor.is_visible());

    corridor.set_corridor(1.0850, 1.0820, 1.0860, 1.0890, 3);
    assert(corridor.is_visible());
    assert(corridor.upper_bound() == 1.0890);
    assert(corridor.lower_bound() == 1.0820);

    ViewportTransform vt;
    vt.min_price = 1.0800;
    vt.max_price = 1.0900;
    vt.canvas_height_px = 600;

    auto poly = corridor.compute_polygon(vt, 10);
    assert(poly.visible);
    assert(poly.upper_points_px.size() == 4); // anchor + 3 future steps
    assert(poly.lower_points_px.size() == 4);

    std::cout << "[PASS] test_confidence_corridor" << std::endl;
}

void test_order_canvas_overlay() {
    using namespace clow::ui;
    using namespace clow::risk;

    OrderCanvasOverlay overlay;
    std::vector<ManagedOrder> orders;

    ManagedOrder o1;
    o1.client_id = 1001;
    o1.order_type = "BUY_LIMIT";
    o1.entry_price = 1.0840;
    o1.stop_loss = 1.0820;
    o1.take_profit = 1.0880;
    o1.volume = 0.50;
    o1.state = OrderState::Pending;
    orders.push_back(o1);

    overlay.update_orders(orders);
    assert(overlay.order_count() == 1);

    ViewportTransform vt;
    vt.min_price = 1.0800;
    vt.max_price = 1.0900;
    vt.canvas_height_px = 600;

    auto lines = overlay.compute_lines(vt, 0.0001);
    assert(lines.size() == 3); // ENTRY, SL, TP
    assert(lines[0].line_type == "ENTRY");
    assert(lines[1].line_type == "SL");
    assert(lines[2].line_type == "TP");

    std::cout << "[PASS] test_order_canvas_overlay" << std::endl;
}

void test_chart_rendering_fps_benchmark() {
    using namespace clow::ui;
    using namespace clow::ai;
    using namespace clow::risk;

    ChartSeries series(1000);
    for (int i = 0; i < 500; ++i) {
        series.push_bar(CandleBar{1000 * i * 60, 1.0800 + i * 0.0001, 1.0810 + i * 0.0001, 1.0790 + i * 0.0001, 1.0805 + i * 0.0001, 50.0});
    }

    GhostCandleOverlay ghost;
    ForecasterPrediction pred;
    pred.valid = true;
    pred.direction_prob = 0.75f;
    pred.anatomy = {0.6f, 0.2f, 0.2f, 1.2f};
    pred.quantiles_high = {0.5f, 1.5f, 2.5f};
    pred.quantiles_low = {0.4f, 1.0f, 2.0f};
    ghost.update_forecast(1.0850, 0.0020, pred);

    ConfidenceCorridor corridor;
    corridor.set_corridor(1.0850, 1.0830, 1.0860, 1.0880, 5);

    OrderCanvasOverlay order_lines;
    ManagedOrder ord;
    ord.client_id = 5001;
    ord.entry_price = 1.0845;
    ord.stop_loss = 1.0825;
    ord.take_profit = 1.0885;
    ord.state = OrderState::Pending;
    order_lines.update_orders({ord});

    ViewportTransform vt;
    vt.canvas_width_px = 1920;
    vt.canvas_height_px = 1080;

    // Benchmark 1,000 frames of pipeline geometry calculations
    auto start = std::chrono::high_resolution_clock::now();

    for (int frame = 0; frame < 1000; ++frame) {
        double min_p = 0.0, max_p = 0.0;
        series.compute_viewport_bounds(0, 100, min_p, max_p);
        vt.min_price = min_p;
        vt.max_price = max_p;

        auto g_rect = ghost.compute_render_geometry(vt, 100);
        auto c_poly = corridor.compute_polygon(vt, 100);
        auto o_lines = order_lines.compute_lines(vt, 0.0001);
        (void)g_rect;
        (void)c_poly;
        (void)o_lines;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double per_frame_us = total_us / 1000.0;

    // 60 FPS = 16.6 ms (16,666 us) budget. Chart geometry pipeline must run in < 1,000 us (1ms).
    assert(per_frame_us < 1000.0);

    std::cout << "[PASS] test_chart_rendering_fps_benchmark (Per-frame geometry time: "
              << per_frame_us << " us, FPS capacity: " << static_cast<int>(1000000.0 / per_frame_us)
              << " FPS)" << std::endl;
}
