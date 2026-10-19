"""Eén ingang voor het cockpit-notebook."""
from __future__ import annotations

from pathlib import Path

from .config import laad
from .data import DataLaag
from .modellen import maak_model
from .po import PO
from .reviewer import Reviewer
from .scout import maak_scout
from .workflow import Team, Uitkomst, eerste_blik_tot_split


class Harness:
    def __init__(self, project_yaml: str | Path = "project.yaml"):
        self.config, basis = laad(project_yaml)
        self.data = DataLaag(self.config.laad_data(basis), target=self.config.target)
        self._po: PO | None = None
        model = lambda rol: maak_model(self.config.model(rol), bij_wachten=self._wacht)  # noqa: E731
        self.team = Team(scout=maak_scout(model("data_scout")), reviewer=Reviewer(model("reviewer")))

    def _wacht(self, tekst: str) -> None:
        if self._po is not None:
            self._po.melding(f"Even geduld: {tekst}")

    async def eerste_blik(self, po: PO) -> Uitkomst:
        self._po = po
        return await eerste_blik_tot_split(po, self.data, doel=self.config.doel, taak=self.config.taak, team=self.team)
