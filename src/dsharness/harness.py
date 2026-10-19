"""Eén ingang voor het cockpit-notebook."""
from __future__ import annotations

from pathlib import Path

from .config import laad
from .data import DataLaag
from .modellen import maak_model
from .po import PO
from .prep import maak_prep
from .prep_plan import PrepPlan
from .reviewer import Reviewer
from .scout import maak_scout
from .verkenner import F3Rapport, maak_verkenner
from .workflow import Team, Uitkomst, eerste_blik_tot_split, prepareer, verken_train


class Harness:
    def __init__(self, project_yaml: str | Path = "project.yaml"):
        self.config, basis = laad(project_yaml)
        self.data = DataLaag(self.config.laad_data(basis), target=self.config.target)
        self._po: PO | None = None
        self.po_vragen: list[tuple[str, str]] = []  # gedeeld over de fases
        self.f1: Uitkomst | None = None
        self.f3: F3Rapport | None = None
        self.plan: PrepPlan | None = None
        model = lambda rol: maak_model(self.config.model(rol), bij_wachten=self._wacht)  # noqa: E731
        self.team = Team(scout=maak_scout(model("data_scout")), reviewer=Reviewer(model("reviewer")),
                         verkenner=maak_verkenner(model("verkenner")), prep=maak_prep(model("prep_engineer")))

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

    async def run(self, po: PO) -> None:
        """F1 → F2 → split → F3 → F4. Slaat fases over die al gedaan zijn; stopt als er geen split is."""
        if not self.data.gesplitst:
            await self.eerste_blik(po)
        if not self.data.gesplitst:
            return
        if self.f3 is None:
            await self.verken_train(po)
        await self.prepareer(po)
        po.melding("F4 klaar. Volgende fase: F5 (modelleren met cross-validatie op train).")
