import pytest
from training.validation.run_paper_simulation import PythonPaperSimulation

def test_python_paper_simulation():
    sim = PythonPaperSimulation(symbols=["EURUSD", "GBPUSD", "USDJPY"], initial_balance=50000.0)
    report = sim.run_simulation(ticks_per_pair=200)
    assert report["total_ticks"] == 600
    assert report["signals_evaluated"] == 12
    assert report["trades_executed"] == 12
    assert report["ending_equity"] > report["starting_equity"]
    assert report["zero_gate_breaches"] is True
