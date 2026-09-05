# 🟢 Clow

> **Clow** is a high-performance quantitative trading terminal and time-series AI research engine.

---

## 🏛️ Architecture Overview

Clow is organized as a unified monorepo with strict separation of concerns:

```
clow/
├── training/            # 🐍 Python AI Research Lab (PyTorch, Transformers, Chronos, ONNX)
├── src/                 # ⚡ Pure C++20 / Qt 6 High-Performance Desktop Terminal
│   ├── core/            # Config, Logging, Risk Gates, Order State Machine
│   ├── broker/          # Direct MT5 Windows IPC / DLL bridge (Zero EAs required)
│   ├── ai/              # ONNX Runtime C++ engine (<5ms CPU inference)
│   └── ui/              # OpenGL Candlestick Charts & Ghost-Candle Overlays
├── models/              # 📦 Packaged ONNX models and metadata
├── config/              # Central YAML & JSON configuration files
└── tests/               # Python and C++ unit/integration test suites
```

---

## 🚀 Key Capabilities

1. **Predictive Ghost-Candle Forecasting**: Fine-tuned deep time-series foundation models (Chronos / TSFM) generate probabilistic next-candle quantile boundaries ($10^{\text{th}}$, $50^{\text{th}}$, $90^{\text{th}}$ percentiles).
2. **Tactical Pending Order Policy**: Maps candle forecasts into precise institutional limit orders (`BUY_LIMIT`, `SELL_LIMIT`, dynamic SL/TP, and expiration horizons).
3. **Direct MT5 Windows IPC Bridge**: Zero Expert Advisors (EAs) needed. Direct inter-process communication with MetaTrader 5 for real-time tick streaming and order execution.
4. **Hardware-Accelerated Charting**: 60+ FPS OpenGL charting rendering historical candles, predictive ghost candles, and quantile cones.
5. **Sovereign Risk Governance**: Enforces fractional risk sizing, maximum daily drawdown limits, spread filters, and panic kill-switch liquidation.

---

## 🛠️ Quick Start

### 1. Python Training Environment
```bash
# Install dependencies in editable mode
pip install -e ".[dev]"

# Inspect runtime environment
clow info

# Run Python unit tests
pytest
```

### 2. C++ Qt Desktop Terminal Build
```bash
# Configure and build with CMake (MSVC 2022 / Clang / GCC)
cmake -B build -S .
cmake --build build --config Release

# Run desktop terminal
./build/clow_terminal
```

---

## 📋 Governance & Milestones

All development is milestone-driven and governed strictly by [`AGENTS.md`](./AGENTS.md).  
Track live development progress and roadmap items in [`PROGRESS.md`](./PROGRESS.md).

---

## 📄 License

Licensed under the **Apache License, Version 2.0**.
