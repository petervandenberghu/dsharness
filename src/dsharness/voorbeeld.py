"""Voorbeeldvoorstel voor het prototype (later gemaakt door de Prep Engineer en de Reviewer)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import Bron, ReviewOordeel, Voorstel


def voorbeeld_data(n: int = 2500, seed: int = 1) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    aanrijtijd = r.lognormal(mean=2.2, sigma=0.45, size=n).round(1)
    df = pd.DataFrame({"aanrijtijd": aanrijtijd, "regio": r.choice(["Noord", "Oost", "Zuid", "West"], n)})
    df.loc[r.choice(n, 300, replace=False), "aanrijtijd"] = np.nan
    return df


def voorbeeld_voorstel(oordeel: str = "twijfel") -> Voorstel:
    toelichting = {
        "goed": "Passende keuze bij MCAR en een scheve kolom.",
        "twijfel": "Bij 12% NA verkleint één vaste waarde de spreiding. Overweeg een NA-indicator erbij.",
        "fout": "De mediaan wordt berekend op alle data, dus vóór de split. Dat lekt informatie uit de testset.",
    }[oordeel]
    return Voorstel(
        id="F4-007",
        fase="F4 Preparatie",
        actie="Ontbrekende waarden in 'aanrijtijd' invullen met de mediaan",
        kolommen=["aanrijtijd"],
        observatie="300 NA's in 'aanrijtijd' (12% van de rijen).",
        onderzocht="NA-patroon: rijen met en zonder NA verschillen niet in andere kolommen. Dat wijst op MCAR.",
        po_kennis="De PO weet niet waarom deze waarden ontbreken.",
        plan="Invullen met de mediaan, geleerd op de trainset, als stap in de Pipeline.",
        waarom="De kolom is scheef (rechtsscheef); de mediaan is robuust tegen uitschieters.",
        bronnen=[
            Bron(titel="Van Buuren (2018), FIMD §1.3", url="https://stefvanbuuren.name/fimd/sec-simplesolutions.html"),
            Bron(titel="scikit-learn: SimpleImputer", url="https://scikit-learn.org/stable/modules/impute.html"),
        ],
        alternatief="KNN-imputatie: meer werk, weinig winst bij MCAR.",
        review=ReviewOordeel(
            oordeel=oordeel,
            toelichting=toelichting,
            bronnen=[Bron(titel="scikit-learn: Common pitfalls", url="https://scikit-learn.org/stable/common_pitfalls.html")],
        ),
    )
