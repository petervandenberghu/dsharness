"""Oefendatasets met bekende valkuilen: voor tests, evals en onderwijs.

Elke dataset heeft:
- `valkuilen`: wat een goede verkenning moet vinden (ground truth voor evals);
- `po_kennis`: wat een gesimuleerde PO per kolom weet;
- `juiste_f1`: de juiste vaste regels en split (om F3 los te kunnen testen);
- `na_mechanismen`: het echte NA-mechanisme per kolom na F2 (ground truth voor F3);
- `juiste_prep`: een verdedigbaar F4-plan (om F5 los te kunnen testen);
- `maat`: wat de PO zegt over welke fouten zwaar wegen (businesskennis voor F5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .acties import (Imputeer, NAIndicator, OneHot, OnmogelijkNaarNA, Schaal, Split, VerwijderDubbelen, VerwijderKolom,
                     VerwijderZonderTarget)


@dataclass(frozen=True)
class Valkuil:
    soort: str  # zie Bevinding.soort in agents/scout.py
    kolom: str | None = None


@dataclass
class Oefenset:
    naam: str
    doel: str
    target: str
    taak: str  # "regressie" of "classificatie"
    df: pd.DataFrame
    valkuilen: list[Valkuil]
    po_kennis: dict[str, str] = field(default_factory=dict)
    juiste_split: str = "willekeurig"
    juiste_f1: list = field(default_factory=list)
    na_mechanismen: dict[str, str] = field(default_factory=dict)
    juiste_prep: list = field(default_factory=list)
    maat: str = ""

    def __post_init__(self):
        if self.maat:  # de gesimuleerde PO herkent vragen over de evaluatiemaat aan deze woorden
            for woord in ("maat", "metriek", "evaluatie", "fouten", "weegt", "belangrijk"):
                self.po_kennis.setdefault(woord, self.maat)


def huizen(seed: int = 11) -> Oefenset:
    r = np.random.default_rng(seed)
    n = 58
    wijk = r.choice(["Centrum", "Noord", "Oost", "West"], n)
    eff = pd.Series({"Centrum": 120, "West": 60, "Oost": 30, "Noord": 0})
    af = pd.Series({"Centrum": 1, "West": 4, "Oost": 5, "Noord": 7})
    opp = np.maximum(35, r.normal(100, 28, n).round()).astype(float)
    kamers = np.clip((opp / 28 + r.normal(0, .6, n)).round(), 1, 8)
    bouwjaar = np.where(wijk == "Centrum", 1880 + r.random(n) * 70, 1950 + r.random(n) * 72).round()
    afstand = np.maximum(.3, af[wijk].to_numpy() + r.normal(0, 1.2, n)).round(1)
    prijs = (60 + 3.1 * opp + eff[wijk].to_numpy() - 6 * afstand + .4 * (bouwjaar - 1960) + r.normal(0, 25, n)).round()
    df = pd.DataFrame({"oppervlakte": opp, "kamers": kamers, "bouwjaar": bouwjaar, "wijk": wijk, "afstand_km": afstand, "prijs_k": prijs})
    df.loc[7, ["oppervlakte", "kamers"]] = [410, 8]
    df.loc[7, "prijs_k"] = round(60 + 3.1 * 410 + eff[df.wijk[7]] - 6 * df.afstand_km[7] + .4 * (df.bouwjaar[7] - 1960))
    df.loc[21, "oppervlakte"] = -df.loc[21, "oppervlakte"]
    df.loc[[3, 47], "kamers"] = np.nan
    df.loc[(df.wijk == "Centrum") & (r.random(n) < .5), "bouwjaar"] = np.nan
    df.loc[40, "prijs_k"] = np.nan
    df = pd.concat([df, df.loc[[12, 33]]], ignore_index=True)
    return Oefenset(
        "huizen", "Voorspel de verkoopprijs van een huis (in duizenden euro's).", "prijs_k", "regressie", df,
        [Valkuil("dubbelen"), Valkuil("onmogelijke_waarde", "oppervlakte"), Valkuil("target_na", "prijs_k"),
         Valkuil("na", "kamers"), Valkuil("na", "bouwjaar"), Valkuil("uitschieter", "oppervlakte")],
        {"oppervlakte": "Eén huis staat met een negatieve oppervlakte in het systeem; dat is een tikfout. Het huis van 410 m² is een echte villa.",
         "kamers": "Bij een paar huizen is het aantal kamers vergeten. Daar zit geen reden achter.",
         "bouwjaar": "Van oude panden in het centrum staat het bouwjaar vaak niet in het register.",
         "prijs_k": "Van één huis is de verkoopprijs nooit geregistreerd.",
         "wijk": "Er zijn vier wijken. Het centrum is het duurst."},
        juiste_f1=[VerwijderDubbelen(), OnmogelijkNaarNA(kolom="oppervlakte", regel="<0"), VerwijderZonderTarget(),
                   Split(methode="willekeurig")],
        na_mechanismen={"kamers": "MCAR", "bouwjaar": "MAR", "oppervlakte": "MCAR"},
        juiste_prep=[Imputeer(kolom="kamers", methode="mediaan"), Imputeer(kolom="bouwjaar", methode="knn"),
                     Imputeer(kolom="oppervlakte", methode="mediaan"), OneHot(kolom="wijk"), Schaal()],
        maat="Een gemiddelde fout van een paar tienduizend euro is acceptabel; grote missers zijn erg vervelend.",
    )


def klant(seed: int = 23) -> Oefenset:
    r = np.random.default_rng(seed)
    n = 80
    leeftijd = np.clip(r.normal(42, 13, n).round(), 18, 80)
    inkomen = np.maximum(12, r.normal(42, 15, n).round())
    u = r.random(n)
    contract = np.where(u < .45, "Maand", np.where(u < .8, "Jaar", "Tweejaar"))
    maanden = 1 + r.integers(0, 72, n)
    klachten = np.minimum(6, r.poisson(1.1, n)).astype(float)
    score = 1.8 * (contract == "Maand") + .5 * (contract == "Jaar") + .6 * klachten - .03 * maanden - .07 * (inkomen - 42) + r.normal(0, .5, n)
    opgezegd = np.where(np.argsort(np.argsort(-score)) < 12, "ja", "nee")
    df = pd.DataFrame({"leeftijd": leeftijd, "inkomen_k": inkomen, "contract": contract, "maanden_klant": maanden,
                       "klachten": klachten, "opgezegd": opgezegd})
    df.loc[(df.inkomen_k > 52) & (r.random(n) < .85), "inkomen_k"] = np.nan
    df.loc[[9, 52], "klachten"] = np.nan
    return Oefenset(
        "klant", "Voorspel welke klanten hun abonnement opzeggen.", "opgezegd", "classificatie", df,
        [Valkuil("scheve_target", "opgezegd"), Valkuil("na", "inkomen_k"), Valkuil("na", "klachten")],
        {"inkomen_k": "Inkomen is een vrijwillig veld. Vooral mensen met een hoog inkomen laten het leeg.",
         "klachten": "Door een storing in het klachtensysteem missen bij twee klanten de gegevens.",
         "opgezegd": "Ongeveer één op de zeven klanten zegt op. Die willen we op tijd vinden."},
        juiste_split="gestratificeerd",
        juiste_f1=[Split(methode="gestratificeerd")],
        na_mechanismen={"inkomen_k": "MNAR", "klachten": "MCAR"},
        juiste_prep=[NAIndicator(kolom="inkomen_k"), Imputeer(kolom="inkomen_k", methode="iteratief"),
                     Imputeer(kolom="klachten", methode="mediaan"), OneHot(kolom="contract"), Schaal()],
        maat="Een gemiste opzegger kost ons veel meer dan een klant die we onterecht bellen. Vind er zoveel mogelijk.",
    )


def patient(seed: int = 37) -> Oefenset:
    r = np.random.default_rng(seed)
    p = 24
    leeftijd = np.clip(r.normal(50, 14, p).round(), 21, 80)
    bmi, gl = r.normal(28, 5, p), r.normal(110, 22, p)
    geslacht = np.where(r.random(p) < .5, "M", "V")
    score = .045 * (gl - 110) + .15 * (bmi - 28) + .03 * (leeftijd - 50) + r.normal(0, .5, p)
    diab = np.where(np.argsort(np.argsort(-score)) < 10, "ja", "nee")
    rijen = []
    for i in range(p):
        for b in (1, 2, 3):
            rijen.append({"patient_id": f"P{i + 1:02d}", "bezoek": b, "glucose": round(gl[i] + r.normal(0, 7)),
                          "bmi": round(bmi[i] + r.normal(0, .6), 1), "bloeddruk": round(r.normal(72 + .25 * (leeftijd[i] - 50), 7)),
                          "leeftijd": float(leeftijd[i]), "geslacht": str(geslacht[i]), "diabetes": str(diab[i])})
    df = pd.DataFrame(rijen).astype({"glucose": float, "bloeddruk": float})
    df.loc[[4, 17, 29, 44, 61], "glucose"] = 0
    df.loc[(df.leeftijd < 38) & (r.random(len(df)) < .6), "bloeddruk"] = np.nan
    df.loc[[8, 35, 66], "bmi"] = np.nan
    return Oefenset(
        "patient", "Voorspel of een patiënt diabetes heeft.", "diabetes", "classificatie", df,
        [Valkuil("structuur_groep", "patient_id"), Valkuil("onmogelijke_waarde", "glucose"),
         Valkuil("na", "bloeddruk"), Valkuil("na", "bmi")],
        {"patient_id": "Elke patiënt komt drie keer op controle. Elk bezoek is een aparte rij.",
         "glucose": "Als de glucose niet gemeten is, vult het systeem automatisch 0 in. Een echte waarde van 0 is onmogelijk.",
         "bmi": "Een paar keer is wegen vergeten. Daar zit geen patroon in.",
         "bloeddruk": "Bij jonge patiënten meten we de bloeddruk vaak niet, omdat hun risico laag is.",
         "diabetes": "De diagnose staat vast per patiënt."},
        juiste_split="groep",
        juiste_f1=[OnmogelijkNaarNA(kolom="glucose", regel="=0"), Split(methode="groep", groep_kolom="patient_id")],
        na_mechanismen={"glucose": "MCAR", "bmi": "MCAR", "bloeddruk": "MAR"},
        juiste_prep=[VerwijderKolom(kolom="patient_id"), Imputeer(kolom="glucose", methode="mediaan"),
                     Imputeer(kolom="bmi", methode="mediaan"), Imputeer(kolom="bloeddruk", methode="knn"),
                     OneHot(kolom="geslacht"), Schaal()],
        maat="Het model wordt gebruikt om patiënten te rangschikken voor extra controle; een goede volgorde telt.",
    )


def winkel(seed: int = 53) -> Oefenset:
    r = np.random.default_rng(seed)
    n = 63
    dagen = pd.date_range("2025-03-03", periods=n, freq="D")
    i = np.arange(n)
    temp = (7 + .17 * i + r.normal(0, 2.5, n)).round(1)
    promo = np.where(r.random(n) < .25, "ja", "nee")
    verkoop = (180 + 3.6 * i + 45 * (promo == "ja") + np.where(i % 7 >= 5, 70, 0) + 2.5 * temp + r.normal(0, 15, n)).round()
    df = pd.DataFrame({"datum": dagen.strftime("%Y-%m-%d"), "dag": dagen.strftime("%a"), "temperatuur": temp,
                       "promotie": promo, "verkoop": verkoop})
    df.loc[[11, 28, 49], "temperatuur"] = np.nan
    return Oefenset(
        "winkel", "Voorspel de dagverkoop van een winkel.", "verkoop", "regressie", df,
        [Valkuil("structuur_tijd", "datum"), Valkuil("na", "temperatuur")],
        {"datum": "Eén rij per dag, negen weken achter elkaar. De winkel groeit gestaag.",
         "temperatuur": "De temperatuursensor viel drie keer uit, op willekeurige dagen.",
         "promotie": "Op promotiedagen is er een aanbieding in de folder."},
        juiste_split="tijd",
        juiste_f1=[Split(methode="tijd", tijd_kolom="datum")],
        na_mechanismen={"temperatuur": "MCAR"},
        juiste_prep=[VerwijderKolom(kolom="datum"), Imputeer(kolom="temperatuur", methode="mediaan"), OneHot(kolom="dag"),
                     OneHot(kolom="promotie"), Schaal()],
        maat="Voor de inkoop zijn grote afwijkingen duur; kleine fouten zijn niet erg.",
    )


ALLE = {"huizen": huizen, "klant": klant, "patient": patient, "winkel": winkel}
