"""Gestructureerde data science-acties. De Reviewer toetst deze, de DataLaag voert ze uit."""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class VerwijderDubbelen(BaseModel):
    """Verwijder rijen die in alle kolommen gelijk zijn aan een eerdere rij (vaste regel)."""
    type: Literal["verwijder_dubbelen"] = "verwijder_dubbelen"


class OnmogelijkNaarNA(BaseModel):
    """Zet onmogelijke waarden in een numerieke kolom om naar NA (vaste regel)."""
    type: Literal["onmogelijk_naar_na"] = "onmogelijk_naar_na"
    kolom: str
    regel: Literal["<0", "=0", "<=0"]


class VerwijderZonderTarget(BaseModel):
    """Verwijder rijen waarin de target ontbreekt (vaste regel)."""
    type: Literal["verwijder_zonder_target"] = "verwijder_zonder_target"


class Split(BaseModel):
    """Train/test-split. Vanaf hier is de testset in de kluis."""
    type: Literal["split"] = "split"
    methode: Literal["willekeurig", "gestratificeerd", "tijd", "groep"]
    test_fractie: float = Field(0.25, ge=0.05, le=0.5)
    groep_kolom: str | None = None
    tijd_kolom: str | None = None


Actie = Annotated[Union[VerwijderDubbelen, OnmogelijkNaarNA, VerwijderZonderTarget, Split], Field(discriminator="type")]

VASTE_REGELS = ("verwijder_dubbelen", "onmogelijk_naar_na", "verwijder_zonder_target")


def omschrijf(a: BaseModel) -> str:
    match a:
        case VerwijderDubbelen():
            return "Dubbele rijen verwijderen"
        case OnmogelijkNaarNA(kolom=k, regel=r):
            return f"Waarden {r} in '{k}' omzetten naar NA"
        case VerwijderZonderTarget():
            return "Rijen zonder target verwijderen"
        case Split(methode=m, test_fractie=f):
            return f"Train/test-split: {m}, testset {f:.0%}"
    raise TypeError(a)
