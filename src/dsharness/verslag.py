"""Verslag: alles wat nodig is om het resultaat te reproduceren en te documenteren, als één JSON-bestand.

`reproduceer()` bouwt de hele keten opnieuw op (vaste regels, split, pipeline, CV, eindtest) en controleert
dat dezelfde getallen terugkomen. Zo is het resultaat-notebook ook een verificatie van de code.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from .acties import Actie, CVOpzet, Experiment, PrepActie
from .contracts import Besluit, Voorstel
from .data import DataLaag
from .lab import EindResultaat, Lab, Resultaat, voer_eindtest
from .prep_plan import PrepPlan
from .rapport import Verkenningsrapport
from .verkenner import F3Rapport


class Verslag(BaseModel):
    project: str
    doel: str
    target: str
    taak: str
    data: str
    basis: str = Field(description="Map waartegen een relatief datapad wordt opgelost")
    seed: int
    modellen: dict[str, str | list[str] | None]
    data_policy: str
    gemaakt: datetime = Field(default_factory=datetime.now)
    po_vragen: list[tuple[str, str]] = []
    besluiten: list[tuple[Voorstel, Besluit]] = []
    f1: Verkenningsrapport | None = None
    acties_f1: list[Actie] = []
    f3: F3Rapport | None = None
    prep: list[PrepActie] = []
    vangnet: list[str] = []
    features: list[str] = []
    cv: CVOpzet | None = None
    positief: str | None = None
    experimenten: list[Experiment] = []
    resultaten: list[Resultaat] = []
    gekozen: str | None = None
    eind: EindResultaat | None = None

    def schrijf(self, pad: str | Path) -> None:
        Path(pad).write_text(self.model_dump_json(indent=1), encoding="utf-8")

    @classmethod
    def lees(cls, pad: str | Path) -> "Verslag":
        return cls.model_validate_json(Path(pad).read_text(encoding="utf-8"))


def maak_verslag(c, basis: Path, data: DataLaag, *, besluiten: list[tuple[Voorstel, Besluit]], po_vragen=(),
                 f1: Verkenningsrapport | None = None, f3: F3Rapport | None = None, plan: PrepPlan | None = None,
                 lab: Lab | None = None) -> Verslag:
    df = data.zicht()
    return Verslag(
        project=c.project, doel=c.doel, target=c.target, taak=c.taak, data=c.data, basis=str(Path(basis).resolve()),
        seed=data.seed, modellen=c.modellen, data_policy=c.data_policy, po_vragen=list(po_vragen),
        besluiten=besluiten, f1=f1, acties_f1=data.acties, f3=f3,
        prep=plan.stappen if plan else [], vangnet=plan.vangnet(df) if plan else [],
        features=plan.proef(df)["kolommen"] if plan else [], cv=lab.cv if lab else None,
        positief=lab.positief if lab else None, experimenten=list(lab.experimenten.values()) if lab else [],
        resultaten=list(lab.resultaten.values()) if lab else [], gekozen=lab.gekozen if lab else None,
        eind=lab.eind if lab else None)


@dataclass
class Reproductie:
    data: DataLaag
    plan: PrepPlan
    lab: Lab
    verschillen: list[str]


def _laad(v: Verslag):
    from .config import Config
    cfg = Config(project=v.project, doel=v.doel, target=v.target, taak=v.taak, data=v.data,
                 data_policy="cloud_allowed", modellen={r: None for r in ("data_scout", "reviewer", "verkenner", "prep_engineer", "modeler")})
    return cfg.laad_data(Path(v.basis))


def reproduceer(v: Verslag, *, tolerantie: float = 1e-6) -> Reproductie:
    """Bouw de keten opnieuw op uit het verslag en vergelijk de getallen."""
    data = DataLaag(_laad(v), target=v.target, seed=v.seed)
    for a in v.acties_f1:
        data.voer_uit(a)
    plan = PrepPlan(v.target)
    for s in v.prep:
        plan.voeg_toe(s, data.zicht())
    lab = Lab(data, plan, v.taak, v.positief)
    verschillen: list[str] = []
    if v.cv is None:
        return Reproductie(data, plan, lab, ["geen CV-opzet in het verslag"])
    lab.voer_uit(v.cv)
    oud = {r.naam: r for r in v.resultaten}
    for e in v.experimenten:
        lab.voer_uit(e)
        nieuw, was = lab.resultaten[e.naam], oud.get(e.naam)
        if was is None or abs(nieuw.gemiddelde - was.gemiddelde) > tolerantie:
            verschillen.append(f"experiment '{e.naam}': {was.gemiddelde if was else '-'} → {nieuw.gemiddelde}")
    if v.gekozen:
        from .acties import KiesModel
        lab.voer_uit(KiesModel(experiment=v.gekozen))
    if v.eind is not None:
        e = voer_eindtest(lab, v.gekozen)
        if abs(e.test - v.eind.test) > tolerantie:
            verschillen.append(f"eindtest: {v.eind.test} → {e.test}")
    return Reproductie(data, plan, lab, verschillen)
