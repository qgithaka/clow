# Getting Started with Clow

## System Requirements

- **Operating System:** Windows 10 / 11 (64-bit)
- **Compiler:** MSVC v143+ or GCC 13+ (supporting C++20)
- **Python:** Python 3.11+
- **MetaTrader 5:** Standard desktop MT5 terminal installed

---

## Quickstart

### 1. Build Native C++ Terminal & Core Engine

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### 2. Run Comprehensive Test Suite

```bash
# Run C++ Unit & Benchmark Tests
./build/clow_cpp_tests.exe

# Run Python Research & Pipeline Tests
python -m pytest tests/python -v
```

### 3. Launch Clow Desktop Terminal

```bash
./build/Release/clow_terminal.exe
```

### 4. Train Models & Export ONNX Binaries

```bash
# Train Clow-Forecaster Model
python -m training.models.forecaster --epochs 50

# Train Clow-Tactical-Policy Model
python -m training.models.order_policy --epochs 50

# Export & INT8 Quantize Models to ONNX
python -m training.models.export_onnx --model forecaster --quantize
python -m training.models.export_onnx --model policy --quantize
```
