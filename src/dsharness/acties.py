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


# ---------- F4: stappen in de preparatie-pipeline (geleerd op train, toegepast op alles) ----------
class VerwijderKolom(BaseModel):
    """Gebruik deze kolom niet als feature (bijv. een ID, of vrijwel alles NA)."""
    type: Literal["verwijder_kolom"] = "verwijder_kolom"
    kolom: str


class NAIndicator(BaseModel):
    """Extra 0/1-kolom '<kolom>_ontbreekt'. Nuttig als het ontbreken zelf informatie is (MNAR, soms MAR)."""
    type: Literal["na_indicator"] = "na_indicator"
    kolom: str


class Imputeer(BaseModel):
    """Vul ontbrekende waarden in. knn/iteratief gebruiken andere numerieke kolommen (geschikt bij MAR)."""
    type: Literal["imputeer"] = "imputeer"
    kolom: str
    methode: Literal["gemiddelde", "mediaan", "modus", "constante", "knn", "iteratief"]
    waarde: float | str | None = Field(None, description="Alleen bij 'constante', bijv. 0 of 'onbekend'")


class Afkappen(BaseModel):
    """Kap waarden af op de IQR-grenzen (Tukey), geleerd op train."""
    type: Literal["afkappen"] = "afkappen"
    kolom: str


class LogTransformatie(BaseModel):
    """log(x + 1), voor sterk rechtsscheve kolommen zonder negatieve waarden."""
    type: Literal["log"] = "log"
    kolom: str


class OneHot(BaseModel):
    """Eén 0/1-kolom per categorie; onbekende categorieën in nieuwe data krijgen overal 0."""
    type: Literal["onehot"] = "onehot"
    kolom: str


class TargetEncoding(BaseModel):
    """Vervang een categorie door de gemiddelde target, met cross-fitting (sklearn TargetEncoder)."""
    type: Literal["target_encoding"] = "target_encoding"
    kolom: str


class Schaal(BaseModel):
    """Schaal numerieke kolommen. 'alle' = alle numerieke features (niet de NA-indicatoren)."""
    type: Literal["schaal"] = "schaal"
    kolommen: list[str] | Literal["alle"] = "alle"
    methode: Literal["standaard", "minmax", "robuust"] = "standaard"


# ---------- F5: modelleren met cross-validatie op train ----------
Metriek = Literal["rmse", "mae", "r2", "accuracy", "balanced_accuracy", "f1", "recall", "roc_auc"]
ModelSoort = Literal["baseline", "lineair", "random_forest", "gradient_boosting", "knn"]


class CVOpzet(BaseModel):
    """Hoe er binnen train gevalideerd wordt, en op welke maat modellen worden vergeleken."""
    type: Literal["cv_opzet"] = "cv_opzet"
    methode: Literal["kfold", "stratified", "groep", "tijd"]
    folds: int = Field(5, ge=2, le=20)
    metriek: Metriek


class Experiment(BaseModel):
    """Eén model (met de F4-pipeline ervoor), beoordeeld met cross-validatie op train."""
    type: Literal["experiment"] = "experiment"
    naam: str = Field(description="Korte unieke naam, bijv. 'rf_diep'")
    model: ModelSoort
    parameters: dict[str, float | int | str | None] = Field(default_factory=dict)
    selectie_k: int | None = Field(None, description="Houd de k beste features (binnen elke fold geselecteerd)")
    gebalanceerd: bool = Field(False, description="class_weight='balanced' (alleen classificatie)")


class KiesModel(BaseModel):
    """Kies het experiment dat in F6 één keer op de testset wordt beoordeeld."""
    type: Literal["kies_model"] = "kies_model"
    experiment: str


# ---------- F6: eenmalige eindtest ----------
class Eindtest(BaseModel):
    """Fit het gekozen experiment op heel train en beoordeel het één keer op de testset."""
    type: Literal["eindtest"] = "eindtest"
    experiment: str


Actie = Annotated[
    Union[VerwijderDubbelen, OnmogelijkNaarNA, VerwijderZonderTarget, Split,
          VerwijderKolom, NAIndicator, Imputeer, Afkappen, LogTransformatie, OneHot, TargetEncoding, Schaal,
          CVOpzet, Experiment, KiesModel, Eindtest],
    Field(discriminator="type"),
]
PrepActie = Annotated[
    Union[VerwijderKolom, NAIndicator, Imputeer, Afkappen, LogTransformatie, OneHot, TargetEncoding, Schaal],
    Field(discriminator="type"),
]

VASTE_REGELS = ("verwijder_dubbelen", "onmogelijk_naar_na", "verwijder_zonder_target")
F1_TYPES = (*VASTE_REGELS, "split")
F4_TYPES = ("verwijder_kolom", "na_indicator", "imputeer", "afkappen", "log", "onehot", "target_encoding", "schaal")
F5_TYPES = ("cv_opzet", "experiment", "kies_model")
F6_TYPES = ("eindtest",)


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
        case VerwijderKolom(kolom=k):
            return f"Kolom '{k}' niet als feature gebruiken"
        case NAIndicator(kolom=k):
            return f"NA-indicator voor '{k}'"
        case Imputeer(kolom=k, methode=m, waarde=v):
            return f"'{k}' imputeren met {m}" + (f" ({v})" if m == "constante" else "")
        case Afkappen(kolom=k):
            return f"Uitschieters in '{k}' afkappen (IQR)"
        case LogTransformatie(kolom=k):
            return f"Log-transformatie van '{k}'"
        case OneHot(kolom=k):
            return f"One-hot encoding van '{k}'"
        case TargetEncoding(kolom=k):
            return f"Target encoding van '{k}'"
        case Schaal(kolommen=ks, methode=m):
            return f"Schalen ({m}): {'alle numerieke features' if ks == 'alle' else ', '.join(ks)}"
        case CVOpzet(methode=sc, folds=f, metriek=m):
            return f"Cross-validatie: {sc}, {f} folds, vergelijken op {m}"
        case Experiment(naam=n, model=m, parameters=p, selectie_k=k, gebalanceerd=g):
            extra = [f"{a}={b}" for a, b in p.items()] + ([f"k={k}"] if k else []) + (["gebalanceerd"] if g else [])
            return f"Experiment '{n}': {m}" + (f" ({', '.join(extra)})" if extra else "")
        case KiesModel(experiment=e):
            return f"Model kiezen: '{e}'"
        case Eindtest(experiment=e):
            return f"Eindtest: '{e}' één keer op de testset"
    raise TypeError(a)
