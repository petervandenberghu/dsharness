"""Output van specialisten: bevindingen en conceptvoorstellen (nog zonder Reviewer-oordeel)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .acties import Actie, omschrijf
from .bronnen import BronSleutel, bron
from .contracts import ReviewOordeel, Voorstel

SoortBevinding = Literal[
    "dubbelen", "onmogelijke_waarde", "target_na", "na", "uitschieter",
    "structuur_groep", "structuur_tijd", "scheve_target", "overig",
]


class Bevinding(BaseModel):
    soort: SoortBevinding
    kolom: str | None = None
    beschrijving: str
    bewijs: str = Field(description="Welke verkenning en welke getallen dit laten zien")


class ConceptVoorstel(BaseModel):
    actie: Actie
    observatie: str
    onderzocht: str
    po_kennis: str = Field(description="Wat de PO zei, of dat de PO het niet weet / niet gevraagd")
    plan: str
    waarom: str
    bron_sleutels: list[BronSleutel] = Field(min_length=1)
    alternatief: str | None = None

    def naar_voorstel(self, id: str, fase: str, review: ReviewOordeel) -> Voorstel:
        kolom = getattr(self.actie, "kolom", None) or getattr(self.actie, "groep_kolom", None) or getattr(self.actie, "tijd_kolom", None)
        return Voorstel(
            id=id, fase=fase, actie=omschrijf(self.actie), kolommen=[kolom] if kolom else [],
            observatie=self.observatie, onderzocht=self.onderzocht, po_kennis=self.po_kennis,
            plan=self.plan, waarom=self.waarom, bronnen=[bron(s) for s in self.bron_sleutels],
            alternatief=self.alternatief, review=review, spec=self.actie,
        )


class Verkenningsrapport(BaseModel):
    samenvatting: str
    bevindingen: list[Bevinding]
    voorstellen: list[ConceptVoorstel] = Field(description="Vaste regels (F2) en precies één split, in uitvoervolgorde")

    @model_validator(mode="after")
    def _een_split(self) -> "Verkenningsrapport":
        n = sum(v.actie.type == "split" for v in self.voorstellen)
        if n != 1:
            raise ValueError(f"Het rapport moet precies één split-voorstel bevatten (nu {n}).")
        if self.voorstellen[-1].actie.type != "split":
            raise ValueError("De split moet het laatste voorstel zijn: vaste regels gaan vóór de split.")
        return self
