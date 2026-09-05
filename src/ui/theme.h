#pragma once

#include <string>

namespace clow::ui {

/**
 * @brief Institutional Dark-Mode Color Palette & Stylesheets for Clow Terminal.
 */
struct Theme {
    // Primary Background & Surface Colors
    static constexpr const char* COLOR_BG_DARK = "#0B0E14";        // Deep Obsidian Navy
    static constexpr const char* COLOR_SURFACE_CARD = "#121722";   // Card & Panel Surface
    static constexpr const char* COLOR_SURFACE_HOVER = "#182030";  // Hover state surface
    static constexpr const char* COLOR_BORDER = "#1E2538";         // Subtle separation borders

    // Semantic Accents
    static constexpr const char* COLOR_BULLISH_GREEN = "#00E676";  // Neon Bullish Green
    static constexpr const char* COLOR_BEARISH_RED = "#FF3D71";    // Neon Bearish Red
    static constexpr const char* COLOR_ACCENT_CYAN = "#00B0FF";    // Primary Action Cyan
    static constexpr const char* COLOR_WARNING_AMBER = "#FFB300";  // Cautionary Amber

    // Typography & Status
    static constexpr const char* COLOR_TEXT_PRIMARY = "#FFFFFF";   // Crisp White
    static constexpr const char* COLOR_TEXT_MUTED = "#90A4AE";     // Muted Blue Grey
    static constexpr const char* COLOR_TEXT_DIM = "#546E7A";       // De-emphasized metadata

    /**
     * @brief Generates complete Qt CSS Dark Theme stylesheet.
     */
    static std::string get_dark_stylesheet() {
        return R"(
            QMainWindow {
                background-color: #0B0E14;
                color: #FFFFFF;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            }
            QWidget#centralWidget {
                background-color: #0B0E14;
            }
            QFrame#cardFrame {
                background-color: #121722;
                border: 1px solid #1E2538;
                border-radius: 6px;
            }
            QLabel {
                color: #FFFFFF;
                font-size: 13px;
            }
            QLabel#mutedLabel {
                color: #90A4AE;
                font-size: 11px;
            }
            QPushButton {
                background-color: #182030;
                color: #FFFFFF;
                border: 1px solid #1E2538;
                border-radius: 4px;
                padding: 6px 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1E2538;
                border: 1px solid #00B0FF;
            }
            QPushButton#panicButton {
                background-color: #FF3D71;
                color: #FFFFFF;
                border: 1px solid #FF1744;
                font-weight: 800;
            }
            QPushButton#panicButton:hover {
                background-color: #D50000;
            }
            QComboBox {
                background-color: #121722;
                color: #FFFFFF;
                border: 1px solid #1E2538;
                border-radius: 4px;
                padding: 4px 10px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #1E2538;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #00B0FF;
                border: 1px solid #00B0FF;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
        )";
    }
};

} // namespace clow::ui
