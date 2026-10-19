"""Beslislog: elk voorstel met het besluit van de PO, als JSONL."""
from __future__ import annotations

import json
from pathlib import Path

from .contracts import Besluit, Voorstel


class Beslislog:
    def __init__(self, pad: str | Path = "beslislog.jsonl"):
        self.pad = Path(pad)

    def schrijf(self, voorstel: Voorstel, besluit: Besluit) -> None:
        regel = {"voorstel": voorstel.model_dump(mode="json"), "besluit": besluit.model_dump(mode="json")}
        with self.pad.open("a", encoding="utf-8") as f:
            f.write(json.dumps(regel, ensure_ascii=False) + "\n")

    def lees(self) -> list[tuple[Voorstel, Besluit]]:
        if not self.pad.exists():
            return []
        out = []
        for line in self.pad.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            out.append((Voorstel.model_validate(d["voorstel"]), Besluit.model_validate(d["besluit"])))
        return out
