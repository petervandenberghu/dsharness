"""Beslislog: elk voorstel met het besluit van de PO, en elke vraag aan de PO, als JSONL."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .contracts import Besluit, Voorstel


class Beslislog:
    """Zonder pad krijgt elke run een eigen bestand: logs/beslislog_<datum>_<tijd>.jsonl. Oude runs blijven bewaard."""

    def __init__(self, pad: str | Path | None = None, *, map: str | Path = "logs"):
        if pad is None:
            pad = Path(map) / f"beslislog_{datetime.now():%Y-%m-%d_%H%M%S}.jsonl"
        self.pad = Path(pad)

    def _schrijf(self, regel: dict) -> None:
        self.pad.parent.mkdir(parents=True, exist_ok=True)
        with self.pad.open("a", encoding="utf-8") as f:
            f.write(json.dumps(regel, ensure_ascii=False) + "\n")

    def schrijf(self, voorstel: Voorstel, besluit: Besluit) -> None:
        self._schrijf({"soort": "besluit", "voorstel": voorstel.model_dump(mode="json"), "besluit": besluit.model_dump(mode="json")})

    def schrijf_vraag(self, vraag: str, antwoord: str) -> None:
        self._schrijf({"soort": "vraag", "vraag": vraag, "antwoord": antwoord, "tijdstip": datetime.now().isoformat()})

    def schrijf_rapport(self, fase: str, rapport) -> None:
        self._schrijf({"soort": "rapport", "fase": fase, "rapport": rapport.model_dump(mode="json"), "tijdstip": datetime.now().isoformat()})

    def lees_rapporten(self) -> list[tuple[str, dict]]:
        return [(d["fase"], d["rapport"]) for d in self._regels("rapport")]

    def _regels(self, soort: str) -> list[dict]:
        if not self.pad.exists():
            return []
        regels = (json.loads(r) for r in self.pad.read_text(encoding="utf-8").splitlines())
        return [d for d in regels if d.get("soort", "besluit") == soort]

    def lees(self) -> list[tuple[Voorstel, Besluit]]:
        return [(Voorstel.model_validate(d["voorstel"]), Besluit.model_validate(d["besluit"])) for d in self._regels("besluit")]

    def lees_vragen(self) -> list[tuple[str, str]]:
        return [(d["vraag"], d["antwoord"]) for d in self._regels("vraag")]
