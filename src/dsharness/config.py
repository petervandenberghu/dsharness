"""Projectconfiguratie (project.yaml): doel, data, modellen per rol en de privacyvlag."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, model_validator

LOKALE_PROVIDERS = {"ollama", "vllm", "test", "function"}
ROLLEN = ("data_scout", "reviewer", "verkenner", "prep_engineer", "modeler")


class Config(BaseModel):
    project: str
    doel: str
    target: str
    taak: Literal["regressie", "classificatie"]
    data: str  # pad naar een csv, of "oefenset:<naam>"
    positief: str | None = None  # classificatie: de klasse die telt (standaard de kleinste)
    data_policy: Literal["local_only", "cloud_allowed"] = "local_only"
    modellen: dict[str, str | list[str] | None]

    @model_validator(mode="after")
    def _controleer(self) -> "Config":
        ontbreekt = [r for r in ROLLEN if r not in self.modellen]
        if ontbreekt:
            raise ValueError(f"Geen model ingesteld voor: {', '.join(ontbreekt)}")
        if self.data_policy == "local_only":
            cloud = {r: n for r, m in self.modellen.items() for n in ([m] if isinstance(m, str) else m or [])
                     if n.split(":", 1)[0] not in LOKALE_PROVIDERS}
            if cloud:
                raise ValueError(f"data_policy is local_only, maar deze rollen gebruiken een cloudmodel: {cloud}")
        return self

    def model(self, rol: str) -> str | list[str] | None:
        return self.modellen[rol]

    def laad_data(self, basis: Path) -> pd.DataFrame:
        if self.data.startswith("oefenset:"):
            from .datasets import ALLE
            return ALLE[self.data.split(":", 1)[1]]().df
        return pd.read_csv(basis / self.data)


def laad(pad: str | Path) -> tuple[Config, Path]:
    pad = Path(pad)
    return Config.model_validate(yaml.safe_load(pad.read_text(encoding="utf-8"))), pad.parent
