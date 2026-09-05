"""Unit tests for MT5 broker connector."""

from unittest.mock import MagicMock, patch

from training.data.mt5_connector import (
    BrokerAccount,
    MT5BrokerConnector,
    SymbolMetadata,
)


def test_broker_account_model() -> None:
    """Verify BrokerAccount data structure."""
    acc = BrokerAccount(
        login=1234567,
        server="Demo-Server",
        balance=50000.0,
        equity=50250.0,
        currency="USD",
        is_connected=True,
    )
    assert acc.login == 1234567
    assert acc.balance == 50000.0
    assert acc.is_connected is True


def test_symbol_metadata_model() -> None:
    """Verify SymbolMetadata contract specs."""
    sym = SymbolMetadata(
        name="EURUSD",
        digits=5,
        point=0.00001,
        spread=1.2,
    )
    assert sym.name == "EURUSD"
    assert sym.digits == 5


def test_mt5_connector_offline_handling() -> None:
    """Verify connector handles uninitialized state gracefully."""
    conn = MT5BrokerConnector(login=0)
    assert conn.get_account_info() is None
    assert conn.get_symbol_catalog() == []
    assert conn.get_live_tick("EURUSD") is None


def test_mt5_connector_with_mock_mt5() -> None:
    """Verify connector interacts with MT5 API when available."""
    mock_mt5 = MagicMock()
    mock_mt5.initialize.return_value = True

    mock_acc = MagicMock()
    mock_acc.login = 998877
    mock_acc.server = "MetaQuotes-Demo"
    mock_acc.name = "Test User"
    mock_acc.company = "MetaQuotes Ltd."
    mock_acc.currency = "USD"
    mock_acc.balance = 25000.0
    mock_acc.equity = 25100.0
    mock_acc.margin = 500.0
    mock_acc.margin_free = 24600.0
    mock_acc.leverage = 100.0
    mock_acc.trade_mode = 0  # Demo
    mock_acc.connected = True
    mock_mt5.account_info.return_value = mock_acc

    mock_sym = MagicMock()
    mock_sym.name = "EURUSD"
    mock_sym.currency_base = "EUR"
    mock_sym.currency_profit = "USD"
    mock_sym.digits = 5
    mock_sym.point = 0.00001
    mock_sym.spread = 1.1
    mock_sym.trade_contract_size = 100000.0
    mock_sym.volume_min = 0.01
    mock_sym.volume_max = 100.0
    mock_sym.volume_step = 0.01
    mock_sym.description = "Euro vs US Dollar"
    mock_mt5.symbols_get.return_value = [mock_sym]

    mock_tick = MagicMock()
    mock_tick.bid = 1.08500
    mock_tick.ask = 1.08512
    mock_tick.last = 1.08500
    mock_tick.volume = 10.0
    mock_tick.time = 1700000000.0
    mock_mt5.symbol_info_tick.return_value = mock_tick

    with patch("training.data.mt5_connector.mt5", mock_mt5), \
         patch("training.data.mt5_connector.HAS_MT5", True):
        conn = MT5BrokerConnector(login=998877, server="MetaQuotes-Demo")
        assert conn.initialize() is True

        acc = conn.get_account_info()
        assert acc is not None
        assert acc.login == 998877
        assert acc.balance == 25000.0
        assert acc.trade_mode == "Demo"

        catalog = conn.get_symbol_catalog()
        assert len(catalog) == 1
        assert catalog[0].name == "EURUSD"

        tick = conn.get_live_tick("EURUSD")
        assert tick is not None
        assert tick.bid == 1.08500
        assert tick.spread_pips == 1.2

        conn.shutdown()
