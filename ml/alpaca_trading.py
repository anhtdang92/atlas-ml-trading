"""
Alpaca Trading Client for ATLAS Stock ML Intelligence System

Integrates Alpaca paper/live trading with the portfolio rebalancing pipeline.
Uses the Alpaca REST API directly via requests (no SDK dependency).

Features:
- Paper trading by default, explicit opt-in for live trading
- Converts PortfolioRebalancer orders to Alpaca market orders
- Risk validation on every order (max single order = 10% of portfolio)
- Full logging of all order attempts and results
- Loads credentials from config/secrets.yaml or environment variables

API docs: https://docs.alpaca.markets/reference

Author: Anh Dang
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"
API_VERSION = "v2"

MIN_TRADE_SIZE = 100.0  # Minimum trade size in USD (matches PortfolioRebalancer)
MAX_SINGLE_ORDER_PCT = 0.10  # Cannot buy/sell more than 10% of portfolio in one order
REQUEST_TIMEOUT = 30  # seconds


class AlpacaTradingError(Exception):
    """Raised when an Alpaca API call fails."""
    pass


class OrderValidationError(Exception):
    """Raised when an order fails pre-submission risk checks."""
    pass


class AlpacaTradingClient:
    """Client for executing trades via the Alpaca REST API.

    Defaults to paper trading. Live trading requires explicit opt-in
    by setting paper=False AND confirming via the live_trading_confirmed flag.

    Usage:
        # From environment variables
        client = AlpacaTradingClient(
            api_key=os.environ["ALPACA_API_KEY"],
            secret_key=os.environ["ALPACA_SECRET_KEY"],
        )

        # From config file
        client = AlpacaTradingClient.from_config("config/secrets.yaml")

        # Get account info
        account = client.get_account()

        # Execute rebalancing orders from PortfolioRebalancer
        orders = rebalancer.calculate_rebalancing_orders(portfolio_value=25000)
        results = client.execute_rebalancing_orders(orders)
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        live_trading_confirmed: bool = False,
    ) -> None:
        """Initialize the Alpaca trading client.

        Args:
            api_key: Alpaca API key ID.
            secret_key: Alpaca API secret key.
            paper: If True (default), use the paper trading endpoint.
            live_trading_confirmed: Must be True to enable live trading.
                Ignored when paper=True.

        Raises:
            ValueError: If credentials are missing or live trading is
                requested without explicit confirmation.
        """
        if not api_key or not secret_key:
            raise ValueError(
                "Alpaca API key and secret key are required. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables "
                "or provide them in config/secrets.yaml."
            )

        if not paper and not live_trading_confirmed:
            raise ValueError(
                "Live trading requires explicit confirmation. "
                "Pass live_trading_confirmed=True to acknowledge real-money risk."
            )

        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.base_url = PAPER_BASE_URL if paper else LIVE_BASE_URL
        self.api_url = f"{self.base_url}/{API_VERSION}"

        self._session = requests.Session()
        self._session.headers.update({
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        })

        mode = "PAPER" if self.paper else "LIVE"
        logger.info(f"AlpacaTradingClient initialized — mode={mode}, base_url={self.base_url}")

    # ------------------------------------------------------------------
    # Class methods / factories
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config_path: str = "config/secrets.yaml",
        paper: bool = True,
        live_trading_confirmed: bool = False,
    ) -> "AlpacaTradingClient":
        """Create a client from a YAML config file or environment variables.

        Lookup order:
        1. Environment variables ALPACA_API_KEY / ALPACA_SECRET_KEY
        2. config/secrets.yaml → alpaca.api_key / alpaca.api_secret

        Args:
            config_path: Path to the secrets YAML file.
            paper: Use paper trading (default True).
            live_trading_confirmed: Explicit live-trading confirmation.

        Returns:
            An initialized AlpacaTradingClient.

        Raises:
            ValueError: If no credentials are found.
        """
        api_key = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")

        if api_key and secret_key:
            logger.info("Loaded Alpaca credentials from environment variables")
            return cls(
                api_key=api_key,
                secret_key=secret_key,
                paper=paper,
                live_trading_confirmed=live_trading_confirmed,
            )

        # Fall back to YAML config
        secrets_file = Path(config_path)
        if secrets_file.exists():
            try:
                import yaml
            except ImportError:
                try:
                    from ruamel.yaml import YAML as _YAML
                    _yaml_loader = _YAML()

                    def _load_yaml(path: Path) -> dict:
                        with open(path, "r") as fh:
                            return _yaml_loader.load(fh) or {}

                    config = _load_yaml(secrets_file)
                except ImportError:
                    raise ValueError(
                        "Neither PyYAML nor ruamel.yaml is installed. "
                        "Install one with: pip install pyyaml"
                    )
                else:
                    # Already loaded above with ruamel
                    pass
            else:
                with open(secrets_file, "r") as fh:
                    config = yaml.safe_load(fh) or {}

            alpaca_cfg = config.get("alpaca", {})
            api_key = alpaca_cfg.get("api_key", "")
            secret_key = alpaca_cfg.get("api_secret", "")

            if api_key and secret_key:
                logger.info(f"Loaded Alpaca credentials from {config_path}")
                return cls(
                    api_key=api_key,
                    secret_key=secret_key,
                    paper=paper,
                    live_trading_confirmed=live_trading_confirmed,
                )

        raise ValueError(
            "Alpaca credentials not found. Set ALPACA_API_KEY / ALPACA_SECRET_KEY "
            f"environment variables or add them to {config_path}."
        )

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_body: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Send an authenticated request to the Alpaca API.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.).
            endpoint: API path relative to /v2/ (e.g. "account", "orders").
            params: Optional query parameters.
            json_body: Optional JSON request body.

        Returns:
            Parsed JSON response as a dict (or list).

        Raises:
            AlpacaTradingError: On non-2xx responses or network errors.
        """
        url = f"{self.api_url}/{endpoint}"
        try:
            response = self._session.request(
                method=method,
                url=url,
                params=params,
                json=json_body,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            logger.error(f"Alpaca API request failed: {method} {url} — {exc}")
            raise AlpacaTradingError(f"Network error contacting Alpaca: {exc}") from exc

        if response.status_code in (200, 201, 204):
            if response.status_code == 204 or not response.content:
                return {}
            return response.json()

        # Error handling
        try:
            error_body = response.json()
        except ValueError:
            error_body = response.text

        logger.error(
            f"Alpaca API error: {response.status_code} {method} {url} — {error_body}"
        )
        raise AlpacaTradingError(
            f"Alpaca API returned {response.status_code}: {error_body}"
        )

    # ------------------------------------------------------------------
    # Account methods
    # ------------------------------------------------------------------

    def get_account(self) -> Dict[str, Any]:
        """Retrieve the current Alpaca account summary.

        Returns:
            Dict with keys: equity, buying_power, cash, portfolio_value,
            currency, account_number, status, paper (bool).
        """
        raw = self._request("GET", "account")
        return {
            "equity": float(raw.get("equity", 0)),
            "buying_power": float(raw.get("buying_power", 0)),
            "cash": float(raw.get("cash", 0)),
            "portfolio_value": float(raw.get("portfolio_value", 0)),
            "currency": raw.get("currency", "USD"),
            "account_number": raw.get("account_number", ""),
            "status": raw.get("status", ""),
            "paper": self.paper,
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        """Retrieve all open positions.

        Returns:
            List of dicts, each containing: symbol, qty, side, market_value,
            avg_entry_price, current_price, unrealized_pl, unrealized_plpc,
            change_today.
        """
        raw_positions = self._request("GET", "positions")
        if not isinstance(raw_positions, list):
            raw_positions = []

        positions = []
        for pos in raw_positions:
            positions.append({
                "symbol": pos.get("symbol", ""),
                "qty": float(pos.get("qty", 0)),
                "side": pos.get("side", "long"),
                "market_value": float(pos.get("market_value", 0)),
                "avg_entry_price": float(pos.get("avg_entry_price", 0)),
                "current_price": float(pos.get("current_price", 0)),
                "unrealized_pl": float(pos.get("unrealized_pl", 0)),
                "unrealized_plpc": float(pos.get("unrealized_plpc", 0)),
                "change_today": float(pos.get("change_today", 0)),
            })

        return positions

    # ------------------------------------------------------------------
    # Order methods
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        """Place a single order on Alpaca.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            qty: Number of shares (supports fractional).
            side: "buy" or "sell".
            order_type: "market" (default), "limit", "stop", "stop_limit".
            time_in_force: "day" (default), "gtc", "ioc", "fok".

        Returns:
            Order confirmation dict with id, status, symbol, qty, side, type,
            submitted_at, and filled info.

        Raises:
            AlpacaTradingError: If the API rejects the order.
        """
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid order side: {side!r}. Must be 'buy' or 'sell'.")

        order_payload = {
            "symbol": symbol.upper(),
            "qty": str(round(qty, 6)),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }

        logger.info(
            f"Placing order: {side.upper()} {qty:.4f} shares of {symbol} "
            f"(type={order_type}, tif={time_in_force}, paper={self.paper})"
        )

        raw = self._request("POST", "orders", json_body=order_payload)

        result = {
            "id": raw.get("id", ""),
            "client_order_id": raw.get("client_order_id", ""),
            "symbol": raw.get("symbol", symbol.upper()),
            "qty": raw.get("qty", str(qty)),
            "side": raw.get("side", side),
            "type": raw.get("type", order_type),
            "status": raw.get("status", "unknown"),
            "submitted_at": raw.get("submitted_at", ""),
            "filled_at": raw.get("filled_at"),
            "filled_qty": raw.get("filled_qty"),
            "filled_avg_price": raw.get("filled_avg_price"),
            "paper": self.paper,
        }

        logger.info(
            f"Order submitted: {result['id']} — {result['symbol']} "
            f"{result['side']} {result['qty']} — status={result['status']}"
        )
        return result

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Check the status of a previously submitted order.

        Args:
            order_id: The Alpaca order ID returned by place_order().

        Returns:
            Dict with id, symbol, side, qty, status, filled_qty,
            filled_avg_price, submitted_at, filled_at.
        """
        raw = self._request("GET", f"orders/{order_id}")
        return {
            "id": raw.get("id", order_id),
            "symbol": raw.get("symbol", ""),
            "side": raw.get("side", ""),
            "qty": raw.get("qty", ""),
            "status": raw.get("status", "unknown"),
            "filled_qty": raw.get("filled_qty"),
            "filled_avg_price": raw.get("filled_avg_price"),
            "submitted_at": raw.get("submitted_at", ""),
            "filled_at": raw.get("filled_at"),
        }

    # ------------------------------------------------------------------
    # Rebalancing integration
    # ------------------------------------------------------------------

    def execute_rebalancing_orders(
        self,
        orders: List[Dict],
        current_prices: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a list of rebalancing orders from PortfolioRebalancer.

        Takes orders in the format produced by
        ``PortfolioRebalancer.calculate_rebalancing_orders()`` and converts
        them into Alpaca market orders.

        Each input order dict is expected to have at least:
            - symbol (str)
            - type ("BUY" or "SELL")
            - amount_usd (float)

        Args:
            orders: List of order dicts from the rebalancer.
            current_prices: Optional dict of {symbol: price}. If not provided,
                the method will fetch the latest quote from Alpaca for each
                symbol to determine share quantities.

        Returns:
            List of result dicts, one per input order, each containing the
            original order info plus execution results or error details.
        """
        if not orders:
            logger.info("No rebalancing orders to execute.")
            return []

        # Fetch account to get portfolio value for risk validation
        try:
            account = self.get_account()
            portfolio_value = account.get("portfolio_value", 0)
        except AlpacaTradingError:
            logger.error("Could not fetch account — aborting rebalancing execution")
            return [
                {**order, "status": "error", "error": "Failed to fetch account info"}
                for order in orders
            ]

        results = []
        for order in orders:
            symbol = order.get("symbol", "")
            order_side = order.get("type", "").upper()
            amount_usd = order.get("amount_usd", 0.0)

            result = {
                "symbol": symbol,
                "side": order_side,
                "amount_usd": amount_usd,
                "paper": self.paper,
                "timestamp": datetime.now().isoformat(),
            }

            # --- Validate ---
            validation_error = self._validate_order(
                symbol=symbol,
                side=order_side,
                amount_usd=amount_usd,
                portfolio_value=portfolio_value,
            )
            if validation_error:
                result["status"] = "rejected"
                result["error"] = validation_error
                logger.warning(f"Order rejected: {symbol} {order_side} ${amount_usd:.2f} — {validation_error}")
                results.append(result)
                continue

            # --- Determine share quantity ---
            price = (current_prices or {}).get(symbol)
            if price is None:
                try:
                    price = self._get_latest_price(symbol)
                except AlpacaTradingError as exc:
                    result["status"] = "error"
                    result["error"] = f"Could not fetch price for {symbol}: {exc}"
                    logger.error(result["error"])
                    results.append(result)
                    continue

            if price <= 0:
                result["status"] = "error"
                result["error"] = f"Invalid price for {symbol}: {price}"
                results.append(result)
                continue

            qty = round(amount_usd / price, 6)  # fractional shares

            # --- Submit order ---
            side = "buy" if order_side == "BUY" else "sell"
            try:
                order_result = self.place_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    order_type="market",
                )
                result.update({
                    "status": order_result.get("status", "submitted"),
                    "order_id": order_result.get("id", ""),
                    "qty": qty,
                    "estimated_price": price,
                    "alpaca_response": order_result,
                })
            except AlpacaTradingError as exc:
                result["status"] = "error"
                result["error"] = str(exc)
                logger.error(f"Order execution failed: {symbol} {side} {qty} — {exc}")

            results.append(result)

        # Summary log
        submitted = sum(1 for r in results if r["status"] not in ("rejected", "error"))
        rejected = sum(1 for r in results if r["status"] == "rejected")
        errored = sum(1 for r in results if r["status"] == "error")
        logger.info(
            f"Rebalancing execution complete: {submitted} submitted, "
            f"{rejected} rejected, {errored} errors (paper={self.paper})"
        )

        return results

    # ------------------------------------------------------------------
    # Risk validation
    # ------------------------------------------------------------------

    def _validate_order(
        self,
        symbol: str,
        side: str,
        amount_usd: float,
        portfolio_value: float,
    ) -> Optional[str]:
        """Validate an order against risk limits before submission.

        Args:
            symbol: Ticker symbol.
            side: "BUY" or "SELL".
            amount_usd: Dollar amount of the order.
            portfolio_value: Current total portfolio value.

        Returns:
            None if the order passes validation, or a string describing
            the reason for rejection.
        """
        if not symbol or not symbol.strip():
            return "Missing or empty symbol"

        if side not in ("BUY", "SELL"):
            return f"Invalid order side: {side!r}"

        if amount_usd < MIN_TRADE_SIZE:
            return (
                f"Order size ${amount_usd:.2f} is below minimum "
                f"trade size ${MIN_TRADE_SIZE:.2f}"
            )

        if portfolio_value > 0:
            order_pct = amount_usd / portfolio_value
            if order_pct > MAX_SINGLE_ORDER_PCT:
                return (
                    f"Order ${amount_usd:.2f} is {order_pct:.1%} of portfolio "
                    f"(${portfolio_value:,.2f}), exceeding the "
                    f"{MAX_SINGLE_ORDER_PCT:.0%} single-order limit"
                )

        return None

    # ------------------------------------------------------------------
    # Price helpers
    # ------------------------------------------------------------------

    def _get_latest_price(self, symbol: str) -> float:
        """Fetch the latest trade price for a symbol from Alpaca market data.

        Uses the /v2/stocks/{symbol}/trades/latest endpoint on the
        Alpaca data API.

        Args:
            symbol: Ticker symbol.

        Returns:
            Latest trade price as a float.

        Raises:
            AlpacaTradingError: If the price cannot be fetched.
        """
        data_url = "https://data.alpaca.markets/v2"
        url = f"{data_url}/stocks/{symbol.upper()}/trades/latest"

        try:
            response = self._session.get(url, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            raise AlpacaTradingError(
                f"Network error fetching price for {symbol}: {exc}"
            ) from exc

        if response.status_code != 200:
            raise AlpacaTradingError(
                f"Failed to fetch price for {symbol}: "
                f"HTTP {response.status_code} — {response.text}"
            )

        data = response.json()
        trade = data.get("trade", {})
        price = float(trade.get("p", 0))

        if price <= 0:
            raise AlpacaTradingError(f"Got invalid price {price} for {symbol}")

        return price

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def cancel_all_orders(self) -> Dict[str, Any]:
        """Cancel all open orders.

        Returns:
            Dict with count of cancelled orders and status.
        """
        logger.warning(f"Cancelling ALL open orders (paper={self.paper})")
        result = self._request("DELETE", "orders")
        if isinstance(result, list):
            logger.info(f"Cancelled {len(result)} open orders")
            return {"cancelled": len(result), "orders": result}
        return {"cancelled": 0, "response": result}

    def close_all_positions(self) -> Dict[str, Any]:
        """Liquidate all open positions.

        WARNING: This will sell every position in the account.

        Returns:
            Dict with liquidation results.
        """
        logger.warning(f"Closing ALL positions (paper={self.paper})")
        result = self._request("DELETE", "positions")
        if isinstance(result, list):
            logger.info(f"Closed {len(result)} positions")
            return {"closed": len(result), "orders": result}
        return {"closed": 0, "response": result}

    def is_market_open(self) -> bool:
        """Check if the US stock market is currently open.

        Returns:
            True if the market is open for trading.
        """
        try:
            clock = self._request("GET", "clock")
            return clock.get("is_open", False)
        except AlpacaTradingError:
            logger.warning("Could not check market clock — assuming closed")
            return False

    def __repr__(self) -> str:
        mode = "paper" if self.paper else "LIVE"
        return f"AlpacaTradingClient(mode={mode}, base_url={self.base_url})"
