"""Mock currency converter. Fixed canned rates - no real API/network call,
just simulated latency, so the demo never depends on external services."""

from __future__ import annotations

import asyncio

from orchestrator.manifests import ScopeClass
from orchestrator.registry import tool_def

# Rates expressed as "1 USD buys this many units of X".
_USD_RATES = {
    "usd": 1.0,
    "jpy": 157.0,
    "eur": 0.92,
    "gbp": 0.79,
    "inr": 83.5,
    "cad": 1.36,
}


class UnknownCurrencyError(ValueError):
    pass


def _rate(currency: str) -> float:
    key = currency.strip().lower()
    if key not in _USD_RATES:
        raise UnknownCurrencyError(f"unknown currency '{currency}'")
    return _USD_RATES[key]


@tool_def(
    name="fx",
    description="Currency converter (canned/mock exchange rates).",
    capabilities=["currency.convert"],
    scope=ScopeClass.READ,
    priority=1,
    timeout_s=5.0,
    param_schema={
        "amount": "numeric amount to convert",
        "from_currency": "3-letter source currency code, e.g. 'USD'",
        "to_currency": "3-letter target currency code, e.g. 'JPY'",
    },
)
async def fx(amount: float, from_currency: str, to_currency: str) -> dict:
    await asyncio.sleep(0.15)
    try:
        usd_amount = float(amount) / _rate(from_currency)
        converted = usd_amount * _rate(to_currency)
    except UnknownCurrencyError as exc:
        return {"error": str(exc), "amount": amount, "from": from_currency, "to": to_currency}
    return {
        "amount": amount,
        "from": from_currency.upper(),
        "to": to_currency.upper(),
        "converted": round(converted, 2),
        "rate": round(_rate(to_currency) / _rate(from_currency), 6),
    }
