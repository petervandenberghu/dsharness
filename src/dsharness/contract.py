"""Datacontract: wat het model als invoer verwacht, vastgelegd uit de trainset.

Fouten (invoer weigeren): ontbrekende kolom, verkeerd type.
Waarschuwingen (wel voorspellen): onbekende categorie, getal buiten het bereik van train, ontbrekende waarde.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel


class Kolom(BaseModel):
    naam: str
    type: str  # "getal" of "categorie"
    min: float | None = None
    max: float | None = None
    categorieen: list[str] = []
    na_in_train: bool = False


class Contract(BaseModel):
    target: str
    taak: str
    kolommen: list[Kolom]

    def schrijf(self, pad: str | Path) -> None:
        Path(pad).write_text(self.model_dump_json(indent=1), encoding="utf-8")

    @classmethod
    def lees(cls, pad: str | Path) -> "Contract":
        return cls.model_validate_json(Path(pad).read_text(encoding="utf-8"))

    def controleer(self, rijen: list[dict]) -> tuple[list[str], list[str]]:
        """Geeft (fouten, waarschuwingen). Bij fouten mag er niet voorspeld worden."""
        fouten, waarschuwingen = [], []
        for i, rij in enumerate(rijen):
            for k in self.kolommen:
                if k.naam not in rij:
                    fouten.append(f"rij {i}: kolom '{k.naam}' ontbreekt")
                    continue
                w = rij[k.naam]
                if w is None or (isinstance(w, float) and pd.isna(w)):
                    waarschuwingen.append(f"rij {i}: '{k.naam}' ontbreekt (wordt ingevuld door de pipeline)")
                elif k.type == "getal":
                    if isinstance(w, bool) or not isinstance(w, (int, float)):
                        fouten.append(f"rij {i}: '{k.naam}' moet een getal zijn, kreeg {w!r}")
                    elif (k.min is not None and w < k.min) or (k.max is not None and w > k.max):
                        waarschuwingen.append(f"rij {i}: '{k.naam}' = {w} valt buiten het bereik van train [{k.min}, {k.max}]")
                elif str(w) not in k.categorieen:
                    waarschuwingen.append(f"rij {i}: '{k.naam}' = {w!r} kwam in train niet voor")
            if extra := sorted(set(rij) - {k.naam for k in self.kolommen} - {self.target}):
                waarschuwingen.append(f"rij {i}: onbekende kolommen genegeerd: {extra}")
        return fouten, waarschuwingen


def maak_contract(train: pd.DataFrame, features: list[str], target: str, taak: str) -> Contract:
    kolommen = []
    for k in features:
        s = train[k]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            kolommen.append(Kolom(naam=k, type="getal", min=float(s.min()), max=float(s.max()), na_in_train=bool(s.isna().any())))
        else:
            kolommen.append(Kolom(naam=k, type="categorie", categorieen=sorted(map(str, s.dropna().unique())),
                                  na_in_train=bool(s.isna().any())))
    return Contract(target=target, taak=taak, kolommen=kolommen)


def voorbeeld(contract: Contract, train: pd.DataFrame, n: int = 3) -> list[dict]:
    """De eerste n trainrijen als JSON-invoer (voor documentatie en de rooktest)."""
    rijen = train[[k.naam for k in contract.kolommen]].head(n)
    return json.loads(rijen.to_json(orient="records"))
