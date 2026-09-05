# PROGRESS.md – Clow Milestone Tracker

**This is the single source of truth for what the AI agent is allowed to work on right now.**

Before starting any work, the agent must have read `AGENTS.md`.  
All permanent rules (branching strategy, commit format, PR generation, how to mark tasks) live in `AGENTS.md`.

---

## M00 – Monorepo Foundation & Build Systems ✅ COMPLETE

**Branch:** `feat/m00-foundation`  
**Status:** Merged into `development`

### Context
Clow is structured as a unified monorepo containing both the Python AI Training Lab (`training/`) and the C++ Qt Desktop Terminal (`src/`). This milestone establishes the complete directory scaffold, build systems, configuration management, logging, and automated test runners.

### Tasks

- [x] Create the complete monorepo directory layout (`training/`, `src/`, `models/`, `tests/`)
- [x] Configure `pyproject.toml` with PyTorch, Transformers, ONNX, MetaTrader5, DuckDB, PyArrow, Ruff, MyPy, and PyTest
- [x] Configure modern `CMakeLists.txt` for C++20, Qt 6, and ONNX Runtime C++ SDK
- [x] Implement central configuration management (YAML + Pydantic for Python, JSON/Settings for C++)
- [x] Implement structured logging with severity levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- [x] Create minimal CLI and test runners for both Python and C++ targets
- [x] Add `.gitignore`, `.env.example`, and baseline `README.md`

---

## M01 – MT5 Windows IPC Bridge & Chunked Data Engine ✅ COMPLETE

**Branch:** `feat/m01-data-engine`  
**Status:** Merged into `development`

### Context
Connects natively to MetaTrader 5 via local Windows IPC / DLL bindings without requiring any Expert Advisors (EAs). Implements chunked time-window pagination (e.g., 1-year slices) to pull decades of historical OHLCV data without hitting broker server row limits, validates data integrity, and stores datasets in immutable Parquet format queried via DuckDB.

### Tasks

- [x] Implement native Windows IPC broker connector to MT5 (account inspection, symbol catalogs, tick quotes)
- [x] Build chunked historical time-window paginator to extract multi-year OHLCV data in slices
- [x] Implement boundary deduplication, chronological UTC sorting, and weekend market gap handling
- [x] Build strict data health validator (detects negative prices, bad spreads, missing intervals, disordered timestamps)
- [x] Implement immutable Parquet storage and DuckDB analytical query layer
- [x] Generate comprehensive data health reports for extracted datasets
- [x] Write unit tests using synthetic clean and corrupt datasets

---

## M02 – Scale-Free Stationary Feature Pipeline ✅ COMPLETE

**Branch:** `feat/m02-feature-pipeline`  
**Status:** Merged into `development`

### Context
Raw price numbers drift over time and cannot be fed to deep neural networks across multiple years or currency pairs. This milestone builds the scale-free, asset-agnostic feature engineering pipeline that transforms raw OHLCV bars into stationary mathematical geometry, volatility ratios, and multi-timeframe context.

### Tasks

- [x] Implement scale-free candle anatomy features (Body Ratio, Upper/Lower Wick Ratios, Range-to-ATR)
- [x] Implement normalized momentum indicators (Rolling Z-score deviations from EMAs, stationary RSI, MACD ratios)
- [x] Implement volatility regime features (Normalized ATR, rolling Bollinger Band width, volatility expansion ratios)
- [x] Implement institutional Forex session features (Asian, London, NY Overlap, London Close session timing masks)
- [x] Implement multi-timeframe hierarchical context aligner (H1/H4 macro trends aligned to M1/M5 without future leakage)
- [x] Implement strictly causal rolling scalers (zero future look-ahead leakage)
- [x] Write unit tests verifying strict stationarity and causality

---

## M03 – Clow-Forecaster Training Lab (Next-Candle Predictor) ✅ COMPLETE

**Branch:** `feat/m03-forecaster-lab`  
**Status:** Merged into `development`

### Context
Implements the deep time-series foundation model pipeline (fine-tuning Chronos / TSFM) to predict the anatomy, directional probability, and probabilistic quantile boundaries (10th, 50th, 90th percentiles) of upcoming candles.

### Tasks

- [x] Implement time-series sliding window dataset generator for Chronos fine-tuning
- [x] Build Chronos fine-tuning pipeline with PyTorch, Hugging Face Transformers, and AutoGluon
- [x] Implement quantile loss / pinball loss evaluation for predictive high/low boundaries
- [x] Implement directional probability classifier ($P_{\text{Bull}}$ vs $P_{\text{Bear}}$)
- [x] Build training, validation, and evaluation loops with early stopping and checkpointing
- [x] Implement sliding window inference benchmark for CPU latency measurement
- [x] Write unit tests for model initialization, training step, and quantile generation

---

## M04 – Clow-Tactical Order Policy Lab ✅ COMPLETE

**Branch:** `feat/m04-tactical-order-lab`  
**Status:** Merged into `development`

### Context
Model 2 takes the candle forecast and historical market context to construct optimal pending orders (`BUY_LIMIT`, `SELL_LIMIT`, entry level, dynamic stop-loss, take-profit, and expiration horizon) using asymmetric Triple-Barrier labeling.

### Tasks

- [x] Implement Triple-Barrier labeling engine (Take-Profit vs Stop-Loss with time expiration horizon)
- [x] Build Tactical Order Policy model (optimizes limit entry discount, SL buffer, and TP targets)
- [x] Implement dynamic order expiration calculator (cancels pending order if unfilled within $N$ bars)
- [x] Implement mathematical expectancy filter ($\text{Expected Value} > 0$ with friction deducted)
- [x] Build end-to-end integration pipeline connecting Model 1 predictions to Model 2 order proposals
- [x] Write unit tests for order parameter validation, barrier outcomes, and expectancy calculation

---

## M05 – Zero-Leakage Validation & Statistical Proof Engine ✅ COMPLETE

**Branch:** `feat/m05-proof-engine`  
**Status:** Merged into `development`

### Context
Subjects all models and strategies to institutional validation standards before any deployment is permitted. Prevents curve-fitting through purged walk-forward cross-validation, synthetic noise permutation tests, and spread/slippage stress testing.

### Tasks

- [x] Implement Purged Walk-Forward Cross-Validation with configurable embargo periods
- [x] Implement Synthetic Noise Permutation Test (proves models collapse to 50% coin-flip on shuffled noise)
- [x] Implement Broker Friction & Spread Shock Simulator (tests profitability under 3x widened spreads and 150ms slippage)
- [x] Implement Monte Carlo 1,000-run trade sequence simulation for worst-case drawdown estimation
- [x] Generate comprehensive Statistical Proof Report (Deflated Sharpe Ratio, Profit Factor, Max Drawdown, Win Rate)
- [x] Write unit tests verifying validation gate rejections on overfitting strategies

---

## M06 – ONNX Model Export & C++ Inference Engine ✅ COMPLETE

**Branch:** `feat/m06-onnx-engine`  
**Status:** Merged into `development`

### Context
Converts trained PyTorch models into optimized, lightweight `.onnx` binaries and implements the native C++ ONNX Runtime inference engine capable of sub-5ms CPU execution with zero Python runtime dependencies.

### Tasks

- [x] Implement PyTorch-to-ONNX conversion and quantization (FP16 / INT8) for Model 1 and Model 2
- [x] Package model metadata, input/output tensor schemas, and feature normalization constants into `model_metadata.json`
- [x] Implement native C++ ONNX Runtime wrapper in `src/ai/onnx_engine.cpp`
- [x] Implement C++ sliding window buffer and SIMD-accelerated feature normalizer
- [x] Benchmark and optimize end-to-end C++ inference latency to $< 5\text{ms}$ on CPU
- [x] Write C++ unit tests verifying ONNX outputs match Python PyTorch outputs within $10^{-5}$ tolerance

---

## M07 – Sovereign Risk Manager & Order State Machine ✅ COMPLETE

**Branch:** `feat/m07-risk-engine`  
**Status:** Merged into `development`

### Context
Implements the sovereign risk governance layer and the native C++ order state machine that controls trade sizing, enforces daily loss limits, manages pending order lifecycles, and executes panic kill-switch commands.

### Tasks

- [x] Implement fractional risk and dynamic Kelly lot sizing engine based on account equity and stop distance
- [x] Implement Sovereign Risk Gates (Daily Max Drawdown limit, Max Open Trades limit, Max Spread filter)
- [x] Implement C++ Pending Order State Machine (PENDING $\to$ FILLED / EXPIRED / REJECTED / CANCELLED)
- [x] Implement instant Panic Kill-Switch (closes all open positions and cancels all pending orders in one call)
- [x] Implement immutable risk audit event logger
- [x] Write unit tests covering risk rejections, state transitions, and panic liquidation

---

## M08 – C++ Qt Modern Desktop Terminal & Navigation ✅ COMPLETE

**Branch:** `feat/m08-desktop-ui`  
**Status:** Merged into `development`

### Context
Builds the core modern dark-mode desktop terminal in C++ Qt 6, featuring responsive multi-viewport layouts, broker profile switching, account status telemetry, and configuration control panels.

### Tasks

- [x] Build Qt 6 MainWindow layout with responsive splitters and dark-mode styling
- [x] Implement Top Telemetry Header (Broker connection status, Account Balance, Equity, Server Ping, Trading Mode)
- [x] Implement Left Control Sidebar (Pair selector, Timeframe selector, Model selector, Mode toggle)
- [x] Implement Broker Account Manager modal (secure credential storage, account switcher, connection status)
- [x] Implement Status & Notification toast system for trade events and errors
- [x] Write UI unit tests verifying state updates, mode toggles, and account switching

---

## M09 – Hardware-Accelerated Charting & Ghost-Candle Overlays ✅ COMPLETE

**Branch:** `feat/m09-charting-engine`  
**Status:** Merged into `development`

### Context
Implements hardware-accelerated 60+ FPS candlestick charting in C++ Qt (OpenGL / QCustomPlot) with real-time MT5 tick streaming and predictive **"Ghost Candle"** overlay rendering showing Model 1 forecasted High, Low, and Close bounds.

### Tasks

- [x] Implement high-performance Candlestick Chart renderer with smooth zoom, pan, and crosshair tools
- [x] Implement real-time bar updates from MT5 tick stream without UI thread stutter
- [x] Implement **Ghost-Candle Predictive Overlay** rendering forecasted upcoming bar geometry
- [x] Implement Quantile Confidence Corridor rendering (10th–90th percentile boundary bands)
- [x] Implement interactive Pending Order level lines (Entry, SL, TP) directly on the chart canvas
- [x] Write chart rendering benchmark tests verifying 60+ FPS performance under rapid tick updates

---

## M10 – Tactical AI Radar & Co-Pilot / Auto-Pilot Dashboard 🔄 IN PROGRESS

**Branch:** `feat/m10-trading-dashboard`  
**Status:** Active – agent is working here

### Context
Integrates the full live trading experience into the desktop terminal, featuring the Tactical AI Radar panel, Co-Pilot manual order approval, Auto-Pilot autonomous trigger loops, and the live MT5 positions/orders dock.

### Tasks

- [ ] Implement Tactical AI Radar panel (displays Model 1 confidence, direction, and Model 2 order proposals)
- [ ] Implement Co-Pilot Mode workflow (renders proposed order parameters with instant [Approve] / [Reject] buttons)
- [ ] Implement Auto-Pilot Mode workflow (autonomously dispatches pending orders to MT5 when Confidence $> 70\%$)
- [ ] Implement Bottom Dock Live Position & Order Table (shows tickets, volumes, entry, floating P/L, and Close buttons)
- [ ] Implement single-click ticket cancellation and position closing via direct MT5 bridge
- [ ] Write integration tests for Co-Pilot approvals and Auto-Pilot trigger dispatches

---

## M11 – End-to-End Simulation, Packaging & Standalone Release

**Branch:** `feat/m11-release-packaging`  
**Status:** Planned

### Context
Performs full end-to-end integration and paper trading simulation, configures automated GitHub Actions CI/CD workflows, and packages the complete Clow desktop terminal with pre-trained ONNX models into a standalone Windows installer.

### Tasks

- [ ] Run full end-to-end paper simulation against live MT5 tick streams across multiple pairs
- [ ] Configure GitHub Actions CI workflow (linting, Python tests, C++ compilation, test suites)
- [ ] Configure GitHub Actions CD release workflow to build standalone Windows `.exe` installer (InnoSetup / CPack)
- [ ] Bundle pre-trained ONNX models and documentation into release packaging
- [ ] Conduct final system audit and security review
