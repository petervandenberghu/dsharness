"""Datacontracten tussen agents, Reviewer en PO."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .acties import Actie


class Bron(BaseModel):
    titel: str
    url: str


class ReviewOordeel(BaseModel):
    oordeel: Literal["goed", "twijfel", "fout"]
    toelichting: str
    bronnen: list[Bron] = []


class Voorstel(BaseModel):
    """Een data science-actie die akkoord van de PO nodig heeft."""

    id: str
    fase: str
    actie: str
    kolommen: list[str] = []
    observatie: str
    onderzocht: str = Field(description="Welk onderzoek is gedaan, bijv. NA-patroon")
    po_kennis: str = Field(description="Wat de PO erover zei, of dat de PO niets weet")
    plan: str
    waarom: str
    bronnen: list[Bron] = Field(min_length=1)
    alternatief: str | None = None
    review: ReviewOordeel
    spec: Actie | None = Field(None, description="De uitvoerbare actie achter dit voorstel")


Keuze = Literal["akkoord", "aanpassen", "afwijzen", "overrulen"]


class Besluit(BaseModel):
    voorstel_id: str
    keuze: Keuze
    motivatie: str = ""
    review_oordeel: Literal["goed", "twijfel", "fout"]
    tijdstip: datetime = Field(default_factory=datetime.now)

    @model_validator(mode="after")
    def _regels(self) -> "Besluit":
        if self.review_oordeel == "fout" and self.keuze == "akkoord":
            raise ValueError("Bij een 'fout' van de Reviewer kan de PO alleen overrulen, aanpassen of afwijzen.")
        if self.keuze == "overrulen" and self.review_oordeel != "fout":
            raise ValueError("Overrulen kan alleen bij een 'fout' van de Reviewer.")
        if self.keuze in ("aanpassen", "overrulen") and not self.motivatie.strip():
            raise ValueError(f"Bij '{self.keuze}' is een motivatie verplicht.")
        return self
