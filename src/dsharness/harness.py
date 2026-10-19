"""Eén ingang voor het cockpit-notebook."""
from __future__ import annotations

from pathlib import Path

from .config import laad
from .data import DataLaag
from .lab import Lab
from .modeler import maak_modeler
from .modellen import maak_model
from .po import PO
from .prep import maak_prep
from .prep_plan import PrepPlan
from .reviewer import Reviewer
from .scout import maak_scout
from .verkenner import F3Rapport, maak_verkenner
from .workflow import Team, Uitkomst, eerste_blik_tot_split, modelleer, prepareer, verken_train


class Harness:
    def __init__(self, project_yaml: str | Path = "project.yaml"):
        self.config, basis = laad(project_yaml)
        self.data = DataLaag(self.config.laad_data(basis), target=self.config.target)
        self._po: PO | None = None
        self.po_vragen: list[tuple[str, str]] = []  # gedeeld over de fases
        self.f1: Uitkomst | None = None
        self.f3: F3Rapport | None = None
        self.plan: PrepPlan | None = None
        self.lab: Lab | None = None
        model = lambda rol: maak_model(self.config.model(rol), bij_wachten=self._wacht)  # noqa: E731
        self.team = Team(scout=maak_scout(model("data_scout")), reviewer=Reviewer(model("reviewer")),
                         verkenner=maak_verkenner(model("verkenner")), prep=maak_prep(model("prep_engineer")),
                         modeler=maak_modeler(model("modeler")))

    def _wacht(self, tekst: str) -> None:
        if self._po is not None:
            self._po.melding(f"Even geduld: {tekst}")

    def _kw(self) -> dict:
        return dict(doel=self.config.doel, taak=self.config.taak, team=self.team, po_vragen=self.po_vragen)

    async def eerste_blik(self, po: PO) -> Uitkomst:
        self._po = po
        self.f1 = await eerste_blik_tot_split(po, self.data, **self._kw())
        return self.f1

    async def verken_train(self, po: PO) -> F3Rapport:
        self._po = po
        self.f3 = await verken_train(po, self.data, **self._kw())
        return self.f3

    async def prepareer(self, po: PO) -> PrepPlan:
        self._po = po
        self.plan = await prepareer(po, self.data, f3=self.f3, plan=self.plan, **self._kw())
        return self.plan

    async def modelleer(self, po: PO) -> Lab:
        self._po = po
        self.lab = await modelleer(po, self.data, plan=self.plan, positief=self.config.positief, lab=self.lab, **self._kw())
        return self.lab

    async def run(self, po: PO) -> None:
        """F1 → F2 → split → F3 → F4 → F5. Slaat fases over die al gedaan zijn; stopt als een fase niet afkomt."""
        if not self.data.gesplitst:
            await self.eerste_blik(po)
        if not self.data.gesplitst:
            return
        if self.f3 is None:
            await self.verken_train(po)
        if self.plan is None or self.plan.problemen(self.data.zicht()):
            await self.prepareer(po)
        if self.plan.problemen(self.data.zicht()):
            return
        if self.lab is None or not self.lab.gekozen:
            await self.modelleer(po)
        if self.lab.gekozen:
            po.melding(f"F5 klaar: '{self.lab.gekozen}' gekozen. Volgende fase: F6 (eenmalige eindtest).")
