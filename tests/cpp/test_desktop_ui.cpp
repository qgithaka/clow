#include <cassert>
#include <iostream>
#include <string>

#include "../../src/ui/account_manager.h"
#include "../../src/ui/main_window.h"
#include "../../src/ui/sidebar.h"
#include "../../src/ui/telemetry_header.h"
#include "../../src/ui/theme.h"
#include "../../src/ui/toast_manager.h"

void test_theme_styling() {
    using namespace clow::ui;
    std::string css = Theme::get_dark_stylesheet();
    assert(!css.empty());
    assert(css.find("#0B0E14") != std::string::npos); // Dark BG
    assert(css.find("#00E676") == std::string::npos); // CSS contains valid classes
    std::cout << "[PASS] test_theme_styling" << std::endl;
}

void test_telemetry_header() {
    using namespace clow::ui;

    TelemetryHeader header;
    TelemetryDisplayData d;
    d.broker_server = "ICMarkets-Live01";
    d.is_connected = true;
    d.ping_ms = 22.5;
    d.balance = 25000.0;
    d.equity = 25400.0;
    d.trading_mode = "LIVE";
    d.daily_drawdown_pct = 0.5;

    header.update_telemetry(d);
    assert(header.data().equity == 25400.0);
    assert(header.get_connection_badge_color() == Theme::COLOR_BULLISH_GREEN);
    assert(header.get_trading_mode_badge_color() == Theme::COLOR_BULLISH_GREEN);

    d.is_connected = false;
    d.trading_mode = "DISABLED";
    header.update_telemetry(d);
    assert(header.get_connection_badge_color() == Theme::COLOR_BEARISH_RED);
    assert(header.get_trading_mode_badge_color() == Theme::COLOR_BEARISH_RED);

    std::cout << "[PASS] test_telemetry_header" << std::endl;
}

void test_sidebar_controller() {
    using namespace clow::ui;

    SidebarController sidebar;
    assert(sidebar.state().active_symbol == "EURUSD");
    assert(sidebar.state().active_timeframe == "M5");

    // Test valid symbol switch
    assert(sidebar.set_symbol("XAUUSD"));
    assert(sidebar.state().active_symbol == "XAUUSD");

    // Test invalid symbol rejection
    assert(!sidebar.set_symbol("INVALID_SYMBOL"));
    assert(sidebar.state().active_symbol == "XAUUSD");

    // Test timeframe switch
    assert(sidebar.set_timeframe("H1"));
    assert(sidebar.state().active_timeframe == "H1");
    assert(!sidebar.set_timeframe("M300")); // Invalid

    // Test risk clamping
    sidebar.set_risk_per_trade(15.0); // Clamped to 5.0
    assert(sidebar.state().risk_per_trade_pct == 5.0);

    sidebar.set_max_open_trades(100); // Clamped to 20
    assert(sidebar.state().max_open_trades == 20);

    std::cout << "[PASS] test_sidebar_controller" << std::endl;
}

void test_account_manager() {
    using namespace clow::ui;

    AccountManager mgr;
    assert(mgr.profiles().size() == 1);
    assert(mgr.active_profile()->profile_id == "default_demo");

    BrokerProfile live_acc;
    live_acc.profile_id = "ic_markets_live";
    live_acc.name = "IC Markets Live Real";
    live_acc.server = "ICMarkets-Live";
    live_acc.login = 2005912;
    live_acc.is_live = true;
    live_acc.balance = 50000.0;
    live_acc.equity = 50000.0;

    mgr.add_profile(live_acc);
    assert(mgr.profiles().size() == 2);

    assert(mgr.select_active_profile("ic_markets_live"));
    assert(mgr.active_profile()->login == 2005912);
    assert(mgr.active_profile()->is_live);

    mgr.update_connection_status("ic_markets_live", true, 50200.0, 50500.0);
    assert(mgr.active_profile()->is_connected);
    assert(mgr.active_profile()->equity == 50500.0);

    std::cout << "[PASS] test_account_manager" << std::endl;
}

void test_toast_manager() {
    using namespace clow::ui;

    ToastManager toasts;
    assert(toasts.active_toasts().empty());

    int64_t t1 = toasts.show_toast(ToastType::Info, "Welcome", "Terminal initialized");
    int64_t t2 = toasts.show_toast(ToastType::Success, "Order Executed", "BUY 0.5 EURUSD @ 1.0850");

    assert(toasts.active_toasts().size() == 2);
    assert(toasts.total_history_count() == 2);

    toasts.dismiss_toast(t1);
    assert(toasts.active_toasts().size() == 1);
    assert(toasts.active_toasts()[0].id == t2);

    toasts.clear_all();
    assert(toasts.active_toasts().empty());
    assert(toasts.total_history_count() == 2);

    std::cout << "[PASS] test_toast_manager" << std::endl;
}

void test_main_window_controller() {
    using namespace clow::ui;

    MainWindowController win;
    
    // Switch pair and verify toast
    assert(win.switch_symbol("GBPUSD"));
    assert(win.sidebar().state().active_symbol == "GBPUSD");
    assert(win.toasts().active_toasts().size() == 1);

    // Switch timeframe
    assert(win.switch_timeframe("M15"));
    assert(win.sidebar().state().active_timeframe == "M15");
    assert(win.toasts().active_toasts().size() == 2);

    // Test equity update and drawdown sync
    win.on_equity_update(9600.0, 10000.0);
    assert(win.telemetry().data().equity == 9600.0);
    assert(win.telemetry().data().daily_drawdown_pct == 4.0);
    assert(win.risk().is_halted()); // 4.0% breached

    // Test Panic Button Click
    auto summary = win.on_panic_button_clicked();
    assert(summary.execution_success);
    assert(win.kill_switch().is_active());
    assert(win.toasts().active_toasts().back().type == ToastType::Critical);

    std::cout << "[PASS] test_main_window_controller" << std::endl;
}
