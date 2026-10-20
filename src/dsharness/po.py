"""De PO: de gebruiker. In het notebook is dat de Cockpit; in tests en evals een ScriptedPO."""
from __future__ import annotations

from typing import Any, Callable, Protocol

from .contracts import Besluit, Voorstel


class PO(Protocol):
    async def vraag_po(self, vraag: str, context: str = "") -> str: ...
    async def vraag_po_meerdere(self, vragen: list[tuple[str, str]]) -> list[str]: ...
    async def vraag_akkoord(self, v: Voorstel, figuur: Any = None) -> Besluit: ...
    def melding(self, tekst: str) -> None: ...
    def toon_rapport(self, rapport: Any) -> None: ...
    def toon_plan(self, plan: Any, proef: dict) -> None: ...
    def toon_resultaten(self, lab: Any) -> None: ...
    def toon_eindtest(self, lab: Any) -> None: ...


def standaard_besluit(v: Voorstel) -> Besluit:
    """Akkoord, behalve bij een 'fout' van de Reviewer: dan afwijzen."""
    keuze = "afwijzen" if v.review.oordeel == "fout" else "akkoord"
    return Besluit(voorstel_id=v.id, keuze=keuze, review_oordeel=v.review.oordeel)


class ScriptedPO:
    """Gesimuleerde PO: beantwoordt vragen uit `kennis` (per kolom) en besluit met `besluit`."""

    def __init__(self, kennis: dict[str, str] | None = None, besluit: Callable[[Voorstel], Besluit] = standaard_besluit):
        self.kennis = kennis or {}
        self.besluit = besluit
        self.vragen: list[tuple[str, str]] = []
        self.kaarten = 0  # aantal vraagkaarten; één kaart kan meerdere deelvragen hebben
        self.voorstellen: list[tuple[Voorstel, Besluit]] = []
        self.meldingen: list[str] = []
        self.rapporten: list[Any] = []
        self.plannen: list[tuple[Any, dict]] = []
        self.resultaten: list[str] = []
        self.eindtesten: list[Any] = []

    async def vraag_po(self, vraag: str, context: str = "") -> str:
        tekst = f"{vraag} {context}".lower()
        hits = [antwoord for kolom, antwoord in self.kennis.items() if kolom.lower() in tekst]
        antwoord = " ".join(dict.fromkeys(hits)) or "Dat weet ik niet."
        self.vragen.append((vraag, antwoord))
        return antwoord

    async def vraag_po_meerdere(self, vragen: list[tuple[str, str]]) -> list[str]:
        self.kaarten += 1
        return [await self.vraag_po(v, c) for v, c in vragen]

    async def vraag_akkoord(self, v: Voorstel, figuur: Any = None) -> Besluit:
        b = self.besluit(v)
        self.voorstellen.append((v, b))
        return b

    def melding(self, tekst: str) -> None:
        self.meldingen.append(tekst)

    def toon_rapport(self, rapport: Any) -> None:
        self.rapporten.append(rapport)

    def toon_plan(self, plan: Any, proef: dict) -> None:
        self.plannen.append((plan, proef))

    def toon_resultaten(self, lab: Any) -> None:
        self.resultaten.append(lab.overzicht())

    def toon_eindtest(self, lab: Any) -> None:
        self.eindtesten.append(lab.eind)
