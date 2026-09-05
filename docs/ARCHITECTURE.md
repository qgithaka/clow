# Clow: Predictive AI Quantitative Trading Terminal & Monorepo

## Institutional Architecture Overview

Clow is a high-performance quantitative trading platform and desktop terminal engineered for institutional precision, microsecond latency execution, and zero-leakage statistical validation.

```
+-----------------------------------------------------------------------------------+
|                              CLOW MONOREPO ARCHITECTURE                           |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [ Python 3.11+ Research & Training Lab ]                                         |
|  +---------------------+  +----------------------+  +--------------------------+  |
|  | Chunked Data Engine |->| Stationary Features  |->| Clow-Forecaster Lab      |  |
|  | Parquet & DuckDB    |  | Log-returns, vol, WSD|  | Time-decay, quantiles    |  |
|  +---------------------+  +----------------------+  +--------------------------+  |
|                                                               |                   |
|  +---------------------+  +----------------------+            v                   |
|  | ONNX INT8 Exporter  |<-| Statistical Proof    |<-+--------------------------+  |
|  | Dynamic batching    |  | DSR, Monte Carlo, WFO|  | Tactical Policy Lab      |  |
|  +---------------------+  +----------------------+  | Half-Kelly, Triple-Bar   |  |
|            |                                        +--------------------------+  |
|            v                                                                      |
|  [ Pure C++20 Qt 6 High-Performance Desktop Terminal ]                            |
|  +---------------------+  +----------------------+  +--------------------------+  |
|  | ONNX Runtime C++    |->| Tactical AI Radar    |->| Sovereign Risk Manager   |  |
|  | CPU sub-5ms SIMD    |  | Multi-timeframe view |  | Kelly Sizing, DD Gates   |  |
|  +---------------------+  +----------------------+  +--------------------------+  |
|                                                               |                   |
|  +---------------------+  +----------------------+            v                   |
|  | MT5 Windows IPC     |<-| Live Positions Dock  |<-+--------------------------+  |
|  | Shared Memory Pipes |  | Real-time Mark-to-Mkt|  | Order State Machine      |  |
|  | Zero EA DLL overhead|  | 1-click execution    |  | Created->Pending->Filled |  |
|  +---------------------+  +----------------------+  +--------------------------+  |
|            |                                                                      |
|            v                                                                      |
|  +-----------------------------------------------------------------------------+  |
|  | Hardware-Accelerated 60+ FPS Charting Engine & Ghost-Candle Predictive Overlay |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## Key Subsystems

### 1. MT5 Windows IPC Bridge (`src/broker/`)
- Direct Windows shared-memory IPC communicating directly with MetaTrader 5 terminal processes.
- Zero EA (Expert Advisor) installation requirements or external DLL dependencies.
- Sub-millisecond tick streaming and single-click order execution.

### 2. Stationary Feature Engine (`training/data/` & `src/ai/`)
- Scale-free stationary feature pipeline preserving volatility regime transitions without forward-looking data leakage.
- Microsecond sliding window buffer in C++ with SIMD-accelerated z-score normalization.

### 3. Native ONNX Runtime Inference Engine (`src/ai/`)
- Pure C++ inference execution delivering p99 latency under $120\,\mu\text{s}$ ($< 5\text{ms}$ hard gate).
- Dynamic INT8 quantization enabling lightweight CPU execution on consumer laptops and dedicated trading rigs.

### 4. Sovereign Risk Manager & Order State Machine (`src/risk/`)
- Fixed fractional risk and dynamic Half-Kelly lot sizing engine.
- Sovereign Risk Gates: Daily Max Drawdown cap, maximum open trades, spread shock filters, and instant emergency Panic Kill-Switch.
- Deterministic asynchronous state machine managing order lifecycles (Created $\to$ Submitted $\to$ Pending $\to$ Filled / Closed).

### 5. Hardware-Accelerated Charting Engine (`src/ui/`)
- 60+ FPS candlestick rendering ($< 60\,\mu\text{s}$ geometry compute time).
- Predictive **Ghost Candle** overlay displaying forecasted High, Low, and Close geometry ahead of the market.
- Quantile Confidence Corridor (10th–90th percentile) and interactive order canvas lines.

### 6. Tactical AI Radar & Co-Pilot / Auto-Pilot Dashboard (`src/ui/`)
- Real-time directional conviction gauges and excursion metrics.
- Co-Pilot mode for human trader verification with one-click `[Approve]` / `[Reject]`.
- Autonomous Auto-Pilot execution loop triggered on conviction $> 70\%$ within risk boundaries.
