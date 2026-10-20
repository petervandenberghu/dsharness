"""Modellen maken uit een naam ('provider:model') of een lijst namen (fallback), met geduld bij tijdelijke fouten.

Werkt voor elke provider: alles gaat via de foutsoorten van Pydantic AI, niet via provider-specifieke clients.
"""
from __future__ import annotations

import asyncio
import os
import re
from typing import Callable

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.wrapper import WrapperModel

TIJDELIJK = {429, 500, 502, 503, 504}


class LimietBereikt(ModelAPIError):
    """Het model vraagt langer te wachten dan zinvol is (bijvoorbeeld een daglimiet). Een fallback neemt het over."""


def wachttijd(e: ModelAPIError, poging: int) -> float:
    """Retry-After-header, anders een hint in de body ('retry in 20s'), anders exponentieel."""
    if not isinstance(e, ModelHTTPError):
        return min(60.0, 2.0 ** (poging + 1))
    if e.headers and (ra := e.headers.get("retry-after")):
        try:
            return float(ra)
        except ValueError:
            pass
    if m := re.search(r"retry in ([\d.]+)\s*s|retryDelay['\"]?:\s*['\"](\d+(?:\.\d+)?)s", str(e.body)):
        return float(m.group(1) or m.group(2)) + 1
    return min(60.0, 2.0 ** poging)


class GeduldigModel(WrapperModel):
    """Probeert een modelverzoek opnieuw bij tijdelijke fouten: rate limit, overbelasting, time-out, verbinding."""

    def __init__(self, wrapped, *, max_pogingen: int = 6, max_wacht: float = 120,
                 bij_wachten: Callable[[str], None] | None = None):
        super().__init__(wrapped)
        self.max_pogingen = max_pogingen
        self.max_wacht = max_wacht
        self.bij_wachten = bij_wachten

    async def request(self, messages, model_settings, model_request_parameters):
        for poging in range(self.max_pogingen):
            try:
                return await self.wrapped.request(messages, model_settings, model_request_parameters)
            except ModelAPIError as e:
                status = getattr(e, "status_code", None)  # None: time-out of verbindingsfout
                if (status is not None and status not in TIJDELIJK) or poging == self.max_pogingen - 1:
                    raise
                t = wachttijd(e, poging)
                if t > self.max_wacht:
                    raise LimietBereikt(self.model_name,
                        f"{self.model_name} vraagt {t / 3600:.1f} uur te wachten (waarschijnlijk een daglimiet). "
                        "Kies een ander model in project.yaml of probeer het later opnieuw.") from e
                if self.bij_wachten:
                    self.bij_wachten(f"{self.model_name}: {status or type(e).__name__}, opnieuw over {t:.0f}s")
                await asyncio.sleep(t)
        raise AssertionError("onbereikbaar")


def korte_fout(e: BaseException) -> str:
    """Eén leesbare regel voor een modelfout, ook als meerdere fallback-modellen faalden."""
    if isinstance(e, BaseExceptionGroup):
        return "; ".join(korte_fout(x) for x in e.exceptions)
    if isinstance(e, LimietBereikt):
        return f"{e.model_name}: daglimiet bereikt"
    if isinstance(e, ModelHTTPError):
        soort = {429: "limiet bereikt", 503: "overbelast"}.get(e.status_code, "fout")
        return f"{e.model_name}: {soort} ({e.status_code})"
    if isinstance(e, ModelAPIError):
        return f"{e.model_name}: {e.message[:80]}"
    return f"{type(e).__name__}: {str(e)[:120]}"


def _een(naam: str, max_pogingen: int, bij_wachten) -> GeduldigModel:
    if naam.startswith("ollama:"):
        os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    return GeduldigModel(naam, max_pogingen=max_pogingen, bij_wachten=bij_wachten)


def maak_model(naam: str | list[str] | Model | None, *, bij_wachten: Callable[[str], None] | None = None) -> Model | None:
    """Eén naam: geduldig model. Een lijst: het eerste model, en bij aanhoudende fouten het volgende."""
    if naam is None or isinstance(naam, Model):
        return naam
    if isinstance(naam, str):
        return _een(naam, 6, bij_wachten)
    if len(naam) == 1:
        return _een(naam[0], 6, bij_wachten)
    return FallbackModel(*(_een(n, 3, bij_wachten) for n in naam))
