"""Native MetaTrader 5 Windows IPC Broker Connector (No EA required)."""

from datetime import datetime, timezone
import logging
from typing import Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("clow.data.mt5")

try:
    import MetaTrader5 as mt5  # type: ignore
    HAS_MT5 = True
except ImportError:
    mt5 = None
    HAS_MT5 = False


class BrokerAccount(BaseModel):
    """Canonical broker account summary."""
    login: int
    server: str
    name: str = ""
    company: str = ""
    currency: str = "USD"
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = 0.0
    leverage: float = 100.0
    trade_mode: str = "Demo"  # Demo, Contest, Real
    is_connected: bool = False


class SymbolMetadata(BaseModel):
    """Canonical symbol metadata and contract specs."""
    name: str
    currency_base: str = "EUR"
    currency_profit: str = "USD"
    digits: int = 5
    point: float = 0.00001
    spread: float = 1.0
    trade_contract_size: float = 100000.0
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    description: str = ""


class LiveTick(BaseModel):
    """Real-time market tick quote."""
    symbol: str
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    bid: float
    ask: float
    last: float = 0.0
    volume: float = 0.0
    spread_pips: float = 0.0


class MT5BrokerConnector:
    """Manages direct IPC communication with local MetaTrader 5 terminal."""

    def __init__(
        self,
        server: str = "MetaQuotes-Demo",
        login: int = 0,
        password: str = "",
        path: str = "",
        timeout_seconds: int = 30,
    ) -> None:
        self.server = server
        self.login = login
        self.password = password
        self.path = path
        self.timeout_seconds = timeout_seconds
        self._is_initialized = False

    def initialize(self) -> bool:
        """Connects directly to the MetaTrader 5 terminal process."""
        if not HAS_MT5:
            logger.warning("MetaTrader5 package is not available on this platform.")
            return False

        init_kwargs: dict[str, Any] = {"timeout": self.timeout_seconds * 1000}
        if self.path:
            init_kwargs["path"] = self.path
        if self.login > 0:
            init_kwargs["login"] = self.login
        if self.password:
            init_kwargs["password"] = self.password
        if self.server:
            init_kwargs["server"] = self.server

        ok = mt5.initialize(**init_kwargs)
        if not ok:
            err = mt5.last_error()
            logger.error("MT5 initialize failed: %s", err)
            self._is_initialized = False
            return False

        self._is_initialized = True
        logger.info("Connected directly to MT5 terminal via Windows IPC.")
        return True

    def shutdown(self) -> None:
        """Shuts down MT5 IPC connection."""
        if HAS_MT5 and self._is_initialized:
            mt5.shutdown()
        self._is_initialized = False

    def get_account_info(self) -> Optional[BrokerAccount]:
        """Retrieves active MT5 account information."""
        if not self._is_initialized or not HAS_MT5:
            return None

        acc = mt5.account_info()
        if acc is None:
            return None

        trade_mode_map = {0: "Demo", 1: "Contest", 2: "Real"}
        mode_str = trade_mode_map.get(getattr(acc, "trade_mode", 0), "Demo")

        return BrokerAccount(
            login=int(getattr(acc, "login", 0)),
            server=str(getattr(acc, "server", "Unknown")),
            name=str(getattr(acc, "name", "")),
            company=str(getattr(acc, "company", "")),
            currency=str(getattr(acc, "currency", "USD")),
            balance=float(getattr(acc, "balance", 0.0)),
            equity=float(getattr(acc, "equity", 0.0)),
            margin=float(getattr(acc, "margin", 0.0)),
            free_margin=float(getattr(acc, "margin_free", 0.0)),
            leverage=float(getattr(acc, "leverage", 100.0)),
            trade_mode=mode_str,
            is_connected=bool(getattr(acc, "connected", False)),
        )

    def get_symbol_catalog(self) -> list[SymbolMetadata]:
        """Returns catalog of tradeable symbols and their specifications."""
        if not self._is_initialized or not HAS_MT5:
            return []

        symbols = mt5.symbols_get()
        if not symbols:
            return []

        catalog: list[SymbolMetadata] = []
        for s in symbols:
            catalog.append(
                SymbolMetadata(
                    name=str(getattr(s, "name", "")),
                    currency_base=str(getattr(s, "currency_base", "")),
                    currency_profit=str(getattr(s, "currency_profit", "")),
                    digits=int(getattr(s, "digits", 5)),
                    point=float(getattr(s, "point", 0.00001)),
                    spread=float(getattr(s, "spread", 0.0)),
                    trade_contract_size=float(getattr(s, "trade_contract_size", 100000.0)),
                    volume_min=float(getattr(s, "volume_min", 0.01)),
                    volume_max=float(getattr(s, "volume_max", 100.0)),
                    volume_step=float(getattr(s, "volume_step", 0.01)),
                    description=str(getattr(s, "description", "")),
                )
            )
        return catalog

    def get_live_tick(self, symbol: str) -> Optional[LiveTick]:
        """Returns latest live tick quote for symbol."""
        if not self._is_initialized or not HAS_MT5:
            return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        bid = float(getattr(tick, "bid", 0.0))
        ask = float(getattr(tick, "ask", 0.0))
        spread_pip = round((ask - bid) * (100.0 if "JPY" in symbol.upper() else 10000.0), 2)
        t_sec = float(getattr(tick, "time", datetime.now(timezone.utc).timestamp()))

        return LiveTick(
            symbol=symbol,
            timestamp_utc=datetime.fromtimestamp(t_sec, tz=timezone.utc),
            bid=bid,
            ask=ask,
            last=float(getattr(tick, "last", 0.0)),
            volume=float(getattr(tick, "volume", 0.0)),
            spread_pips=spread_pip,
        )
