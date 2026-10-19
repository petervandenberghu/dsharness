"""Reviewer: vaste regels in code, plus een optionele LLM-laag.

De LLM mag een oordeel strenger maken, nooit milder: het eindoordeel is het strengste van de twee.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .acties import (F4_TYPES, F5_TYPES, Afkappen, CVOpzet, Experiment, KiesModel, Imputeer, LogTransformatie, NAIndicator, OneHot, OnmogelijkNaarNA, Schaal, Split,
                     TargetEncoding, VerwijderDubbelen, VerwijderKolom, VerwijderZonderTarget)
from .bronnen import BronSleutel, bron, catalogus_tekst
from .contracts import ReviewOordeel
from .data import DataLaag
from .modellen import maak_model
from .lab import HOGER_IS_BETER, LabFout
from .prep_plan import EENVOUDIG, PrepPlan, PrepPlanFout
from .rapport import Bevinding, ConceptVoorstel

Oordeel = Literal["goed", "twijfel", "fout"]
ERNST: dict[str, int] = {"goed": 0, "twijfel": 1, "fout": 2}


@dataclass
class Uitkomst:
    oordeel: Oordeel
    tekst: str
    bronnen: tuple[str, ...] = ()


@dataclass
class ReviewContext:
    data: DataLaag
    taak: str  # "regressie" of "classificatie"
    bevindingen: list[Bevinding] = field(default_factory=list)
    po_vragen: list[tuple[str, str]] = field(default_factory=list)  # (vraag, antwoord), echt gesteld
    profielen: dict[str, Any] = field(default_factory=dict)  # F3: kolom → KolomProfiel
    plan: PrepPlan | None = None  # F4: het plan tot nu toe
    lab: Any = None  # F5: experimenten en resultaten

    def po_gevraagd_over(self, kolom: str) -> bool:
        return any(kolom.lower() in (v + " " + a).lower() for v, a in self.po_vragen)


# ---------- structuur herkennen (onafhankelijk van wat de specialist zegt) ----------
def groep_kandidaten(ctx: ReviewContext) -> set[str]:
    df = ctx.data.zicht()
    uit = {b.kolom for b in ctx.bevindingen if b.soort == "structuur_groep" and b.kolom}
    for k in df.columns:
        if k != ctx.data.target and "id" in k.lower() and df[k].duplicated().any():
            uit.add(k)
    return uit


def tijd_kandidaten(ctx: ReviewContext) -> set[str]:
    df = ctx.data.zicht()
    uit = {b.kolom for b in ctx.bevindingen if b.soort == "structuur_tijd" and b.kolom}
    for k in df.columns:
        s = df[k]
        if pd.api.types.is_datetime64_any_dtype(s):
            uit.add(k)
        elif (s.dtype == object or pd.api.types.is_string_dtype(s)) and s.notna().any():
            geparsed = pd.to_datetime(s, errors="coerce", format="mixed")
            if geparsed.notna().mean() >= .9:
                uit.add(k)
    return uit


def minderheid(ctx: ReviewContext) -> float | None:
    if ctx.taak != "classificatie":
        return None
    vc = ctx.data.zicht()[ctx.data.target].value_counts(normalize=True)
    return float(vc.min()) if len(vc) > 1 else None


# ---------- regels ----------
def regels(c: ConceptVoorstel, ctx: ReviewContext) -> list[Uitkomst]:
    a, d = c.actie, ctx.data
    df = d.zicht()
    u: list[Uitkomst] = []
    match a:
        case VerwijderDubbelen():
            if d.gesplitst:
                kruis = d.dubbele_paren_over_sets()
                u.append(Uitkomst("fout", f"{kruis} dubbel paar zit al verdeeld over train en test; de testset is besmet.", ("kapoor", "kaufman"))
                         if kruis else Uitkomst("twijfel", "Dubbelingen verwijder je vóór de split. Nu lekt er toevallig niets.", ("kapoor",)))
            elif not df.duplicated().any():
                u.append(Uitkomst("twijfel", "Er zijn geen dubbele rijen; deze stap doet niets."))
            if not d.onderzocht(("dubbelen",)):
                u.append(Uitkomst("twijfel", "Niet onderzocht: zoek eerst naar dubbele rijen, dan weet je wat je weggooit.", ("crisp",)))
        case OnmogelijkNaarNA(kolom=k, regel=r):
            if k not in df.columns:
                return [Uitkomst("fout", f"Kolom '{k}' bestaat niet.")]
            if not pd.api.types.is_numeric_dtype(df[k]):
                return [Uitkomst("fout", f"'{k}' is geen getal.")]
            s = df[k]
            n = int({"<0": s < 0, "=0": s == 0, "<=0": s <= 0}[r].sum())
            if n == 0:
                u.append(Uitkomst("twijfel", f"Geen enkele waarde in '{k}' voldoet aan '{r}'; deze stap doet niets."))
            if not d.onderzocht(("beschrijf", "uitschieters", "vorm"), k):
                u.append(Uitkomst("twijfel", f"Niet onderzocht: bekijk eerst de samenvatting of uitschieters van '{k}'.", ("crisp",)))
            if not ctx.po_gevraagd_over(k):
                u.append(Uitkomst("twijfel", "Of een waarde echt onmogelijk is, weet de PO het best. Vraag het na.", ("crisp",)))
        case VerwijderZonderTarget():
            if df[d.target].notna().all():
                u.append(Uitkomst("twijfel", "Alle rijen hebben een target; deze stap doet niets."))
        case Split():
            u += _split_regels(a, ctx, df)
    if a.type in F4_TYPES:
        u += _f4_regels(a, ctx, df)
    if a.type in F5_TYPES:
        u += _f5_regels(a, ctx, df)
    return u or [Uitkomst("goed", "Geen bezwaren gevonden in de regels.")]


def _num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _f4_regels(a, ctx: ReviewContext, df: pd.DataFrame) -> list[Uitkomst]:
    """F4: regels uit het stappenplan. Lekken kan hier niet meer: de pipeline wordt alleen op train gefit."""
    plan = ctx.plan or PrepPlan(ctx.data.target)
    try:
        plan._valideer(a, df)
    except PrepPlanFout as e:
        return [Uitkomst("fout", str(e))]
    k = getattr(a, "kolom", None)
    prof = ctx.profielen.get(k) if k else None
    mech = getattr(prof, "na_mechanisme", None)
    s = df[k] if k else None
    u: list[Uitkomst] = []
    match a:
        case Imputeer(methode=m):
            n, frac = int(s.isna().sum()), float(s.isna().mean())
            if n == 0:
                u.append(Uitkomst("twijfel", f"In '{k}' ontbreekt in train niets. Het vangnet (mediaan) dekt onverwachte NA al."))
            if prof is None:
                u.append(Uitkomst("twijfel", f"Er is geen F3-profiel van '{k}': het NA-mechanisme is niet onderzocht.", ("fimd_mech",)))
            if m == "gemiddelde" and _num(s) and abs(s.skew()) > 1:
                u.append(Uitkomst("twijfel", f"'{k}' is scheef ({s.skew():.1f}); de mediaan is robuuster dan het gemiddelde.", ("fimd_simple",)))
            if m in EENVOUDIG and frac > .10:
                u.append(Uitkomst("twijfel", f"Er ontbreekt {frac:.0%}. Eén vaste waarde verkleint de spreiding en verzwakt verbanden.", ("fimd_simple",)))
            if mech == "MAR" and m in EENVOUDIG:
                u.append(Uitkomst("twijfel", "Het ontbreken hangt samen met andere kolommen (MAR). knn of iteratief imputeren gebruikt die kolommen.", ("fimd_mech", "knn")))
            if mech == "MNAR" and not plan._van(k, NAIndicator):
                u.append(Uitkomst("twijfel", "Het ontbreken zegt iets over de waarde zelf (MNAR). Voeg eerst een NA-indicator toe.", ("indicator", "fes_missing")))
            if m == "constante" and _num(s) and not plan._van(k, NAIndicator):
                u.append(Uitkomst("twijfel", "Een vaste getalwaarde lijkt op een echte waarde. Gebruik dit alleen samen met een NA-indicator.", ("indicator",)))
            if m == "modus" and _num(s):
                u.append(Uitkomst("twijfel", "De modus is bedoeld voor categorieën; voor getallen is de mediaan meestal beter.", ("impute",)))
        case NAIndicator():
            if not s.isna().any():
                u.append(Uitkomst("twijfel", f"In '{k}' ontbreekt in train niets; de indicator is dan overal 0."))
            elif mech == "MCAR":
                u.append(Uitkomst("goed", "Mag, maar bij toevallig ontbreken (MCAR) voegt de indicator weinig toe.", ("indicator",)))
        case Afkappen():
            q1, q3 = s.quantile(.25), s.quantile(.75)
            if not ((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).any():
                u.append(Uitkomst("twijfel", f"Volgens de IQR-regel heeft '{k}' in train geen uitschieters; deze stap doet niets.", ("tukey",)))
            if not ctx.data.onderzocht(("uitschieters", "verdeling"), k):
                u.append(Uitkomst("twijfel", f"Niet onderzocht: bekijk eerst de uitschieters of de verdeling van '{k}'.", ("tukey",)))
            elif not ctx.po_gevraagd_over(k):
                u.append(Uitkomst("twijfel", f"Zijn de uitschieters in '{k}' fouten of echte waarden? Vraag het de PO; echte waarden afkappen verandert de data.", ("tukey",)))
        case LogTransformatie():
            if abs(s.skew()) < 1:
                u.append(Uitkomst("twijfel", f"'{k}' is nauwelijks scheef ({s.skew():.1f}); een log-transformatie maakt het model vooral lastiger uit te leggen.", ("fes_numeric",)))
        case OneHot():
            if s.nunique() > 15:
                u.append(Uitkomst("twijfel", f"'{k}' heeft {s.nunique()} categorieën; one-hot maakt veel kolommen. Target encoding kan beter.", ("te",)))
        case TargetEncoding():
            if s.nunique() <= 5:
                u.append(Uitkomst("twijfel", f"'{k}' heeft maar {s.nunique()} categorieën; one-hot is eenvoudiger en uitlegbaarder.", ("ohe",)))
        case Schaal(kolommen=ks, methode=m):
            doel = ks if isinstance(ks, list) else [c for c in plan.features(df) if _num(df[c])]
            if m == "minmax":
                sterk = [c for c in doel if ((df[c] - df[c].median()).abs() > 3 * (df[c].quantile(.75) - df[c].quantile(.25))).any()]
                if sterk:
                    u.append(Uitkomst("twijfel", f"Sterke uitschieters in {sterk}: min-max drukt de gewone waarden dan samen. Standaard of robuust werkt beter.", ("prep",)))
        case VerwijderKolom():
            id_achtig = "id" in k.lower() or (not _num(s) and s.nunique() > .5 * s.notna().sum())
            if not id_achtig and s.isna().mean() <= .5 and _num(s):
                y = df[ctx.data.target]
                y = y if _num(y) else (y == y.value_counts().index[-1]).astype(float)
                if abs(s.corr(y)) > .3:
                    u.append(Uitkomst("twijfel", f"'{k}' hangt samen met de target (r = {s.corr(y):.2f}). Weet je zeker dat je hem weg wilt?", ("esl",)))
    return u


def _split_regels(a: Split, ctx: ReviewContext, df: pd.DataFrame) -> list[Uitkomst]:
    d, u = ctx.data, []
    if d.gesplitst:
        return [Uitkomst("fout", "De data is al gesplitst. Eén split is genoeg.")]
    groepen, tijden = groep_kandidaten(ctx), tijd_kandidaten(ctx)
    if groepen and a.methode != "groep":
        u.append(Uitkomst("fout", f"Meerdere rijen per {', '.join(sorted(groepen))}. Splits per groep, anders zit dezelfde eenheid in train én test.", ("cv", "kapoor")))
    if tijden and a.methode != "tijd" and not groepen:
        u.append(Uitkomst("fout", f"'{', '.join(sorted(tijden))}' is een tijdkolom. Splits op tijd: train is het verleden, test de toekomst.", ("fpp", "cv")))
    if a.methode == "groep":
        if not a.groep_kolom:
            u.append(Uitkomst("fout", "Bij een split per groep moet groep_kolom zijn ingevuld."))
        elif not d.onderzocht(("tel_waarden",), a.groep_kolom):
            u.append(Uitkomst("twijfel", f"Niet onderbouwd: tel eerst hoe vaak elke '{a.groep_kolom}' voorkomt.", ("cv",)))
    if a.methode == "tijd":
        if not a.tijd_kolom:
            u.append(Uitkomst("fout", "Bij een split op tijd moet tijd_kolom zijn ingevuld."))
        elif not d.onderzocht(("tijdsverloop",), a.tijd_kolom):
            u.append(Uitkomst("twijfel", f"Niet onderbouwd: bekijk eerst het verloop van de target over '{a.tijd_kolom}'.", ("fpp",)))
    m = minderheid(ctx)
    if m is not None and m < .3 and a.methode == "willekeurig":
        u.append(Uitkomst("twijfel", f"De kleinste klasse is {m:.0%}. Stratificeer, zodat train en test dezelfde verhouding houden.", ("tts",)))
    if a.methode == "gestratificeerd":
        if ctx.taak != "classificatie":
            u.append(Uitkomst("fout", "Stratificeren kan alleen bij classificatie."))
        elif not d.onderzocht(("tel_waarden",), d.target):
            u.append(Uitkomst("twijfel", "Niet onderbouwd: tel eerst de klassen van de target.", ("tts",)))
    if df.duplicated().any():
        u.append(Uitkomst("twijfel", f"Er zitten nog {int(df.duplicated().sum())} dubbele rij(en) in de data. Die kunnen in train én test belanden.", ("kapoor",)))
    if not .15 <= a.test_fractie <= .4:
        u.append(Uitkomst("twijfel", f"Een testset van {a.test_fractie:.0%} is ongebruikelijk; 20 tot 30% is gangbaar.", ("geron",)))
    return u


# ---------- LLM-laag ----------
class LLMOordeel(BaseModel):
    oordeel: Oordeel
    toelichting: str = Field(description="Kort, in eenvoudig Nederlands")
    bron_sleutels: list[BronSleutel] = []


INSTRUCTIES = f"""Je bent de Reviewer in een data science-team. Je beoordeelt één voorstel voordat de product owner (PO) het ziet.
Toets op: data leakage (leert de stap iets van de data vóór de split, of van de testset?), onderbouwing (is het onderzocht,
is de PO gevraagd waar domeinkennis nodig is?) en of de keuze past bij wat er over de data bekend is.
De uitkomst van de vaste regels krijg je mee. Die zijn bindend: je mag strenger zijn, niet milder.
Oordelen: goed (geen bezwaar), twijfel (kan, maar er is een betere optie of het is niet onderbouwd), fout (niet doen).
Schrijf kort en in eenvoudig Nederlands. Citeer alleen bronnen uit deze lijst (sleutel):
{catalogus_tekst()}"""


def maak_llm_reviewer(model) -> Agent[None, LLMOordeel]:
    return Agent(maak_model(model), output_type=LLMOordeel, instructions=INSTRUCTIES, name="reviewer")


class Reviewer:
    def __init__(self, model=None):
        self.llm = maak_llm_reviewer(model) if model is not None else None

    async def beoordeel(self, c: ConceptVoorstel, ctx: ReviewContext) -> ReviewOordeel:
        uitkomsten = regels(c, ctx)
        if self.llm is not None:
            o = (await self.llm.run(self._prompt(c, ctx, uitkomsten))).output
            uitkomsten.append(Uitkomst(o.oordeel, o.toelichting, tuple(o.bron_sleutels)))
        return combineer(uitkomsten)

    @staticmethod
    def _prompt(c: ConceptVoorstel, ctx: ReviewContext, uitkomsten: list[Uitkomst]) -> str:
        d = ctx.data
        verkend = sorted({f"{v.tool}({v.kolom or ''})" for v in d.log})
        return (
            f"Voorstel:\n{c.model_dump_json(indent=1)}\n\n"
            f"Context: taak={ctx.taak}, target={d.target}, gesplitst={d.gesplitst}, rijen in zicht={d.n_train}.\n"
            f"Verkenningen gedaan: {', '.join(verkend) or 'geen'}.\n"
            f"Bevindingen: {[b.model_dump() for b in ctx.bevindingen]}\n"
            f"Vragen aan de PO (echt gesteld): {ctx.po_vragen or 'geen'}\n\n"
            f"Vaste regels: {[(x.oordeel, x.tekst) for x in uitkomsten]}"
        )


def combineer(uitkomsten: list[Uitkomst]) -> ReviewOordeel:
    eind = max(uitkomsten, key=lambda x: ERNST[x.oordeel]).oordeel
    relevant = [x for x in uitkomsten if x.oordeel != "goed"] or uitkomsten
    teksten = list(dict.fromkeys(x.tekst for x in relevant))
    sleutels = list(dict.fromkeys(s for x in relevant for s in x.bronnen))
    return ReviewOordeel(oordeel=eind, toelichting=" ".join(teksten), bronnen=[bron(s) for s in sleutels])


def _f5_regels(a, ctx: ReviewContext, df: pd.DataFrame) -> list[Uitkomst]:
    """F5: past de validatie bij de split, is de maat passend, en is de modelkeuze verdedigbaar?"""
    lab = ctx.lab
    if lab is None:
        return [Uitkomst("fout", "Er is geen lab; F5 kan pas na F4.")]
    try:
        lab.valideer(a)
    except LabFout as e:
        return [Uitkomst("fout", str(e))]
    u: list[Uitkomst] = []
    m = minderheid(ctx)
    match a:
        case CVOpzet(methode=sc, folds=k, metriek=mt):
            split = ctx.data.split_actie.methode if ctx.data.split_actie else None
            nodig = {"groep": "groep", "tijd": "tijd"}.get(split)
            if nodig and sc != nodig:
                u.append(Uitkomst("fout", f"De split was per {split}. Valideer binnen train ook per {nodig}, anders lekt "
                                          "informatie tussen de folds en lijkt het model beter dan het is.", ("cv", "kapoor")))
            if sc in ("groep", "tijd") and not nodig:
                u.append(Uitkomst("fout", f"Een {sc}-schema past niet bij een {split}e split: er is geen {sc}kolom vastgelegd.", ("cv",)))
            if split == "gestratificeerd" and sc == "kfold":
                u.append(Uitkomst("twijfel", "De target is scheef; stratified houdt de klassenverhouding gelijk in elke fold.", ("cv", "santos")))
            if m is not None and m < .3 and mt == "accuracy":
                u.append(Uitkomst("twijfel", f"De kleinste klasse is {m:.0%}. Accuracy beloont dan 'altijd de grootste klasse'. "
                                             "Kies recall, f1, balanced_accuracy of roc_auc.", ("metrics", "santos")))
            if lab.data.n_train / k < 10:
                u.append(Uitkomst("twijfel", f"Met {lab.data.n_train} rijen en {k} folds heeft elke fold minder dan 10 rijen; "
                                             "de scores worden onbetrouwbaar.", ("cv",)))
            woorden = ("maat", "metriek", "evaluatie", "belangrijk", mt)
            if not any(w in (v + " " + an).lower() for v, an in ctx.po_vragen for w in woorden):
                u.append(Uitkomst("twijfel", "De evaluatiemaat is een businesskeuze. Vraag de PO welke fouten het zwaarst wegen "
                                             "(bijv. een gemist geval tegenover een vals alarm).", ("metrics",)))
        case Experiment(model=mod, gebalanceerd=g):
            plan = lab.plan
            if mod in ("lineair", "knn") and not any(isinstance(s, Schaal) for s in plan.stappen):
                u.append(Uitkomst("twijfel", f"{mod} is gevoelig voor de schaal van features; de pipeline schaalt niet.", ("prep",)))
            if g and m is not None and m >= .35:
                u.append(Uitkomst("twijfel", f"De klassen zijn redelijk in balans ({m:.0%}); balanceren is niet nodig.", ("santos",)))
            if len(lab.experimenten) >= 8:
                u.append(Uitkomst("twijfel", f"Al {len(lab.experimenten)} experimenten: hoe meer je vergelijkt, hoe groter de kans "
                                             "dat het beste model vooral geluk had op de validatie.", ("cawley",)))
        case KiesModel(experiment=n):
            r, beste, base = lab.resultaten[n], lab.beste(), lab.baseline()
            hoger = HOGER_IS_BETER.get(lab.cv.metriek, True)
            winst = lambda x, y: (x - y) if hoger else (y - x)  # noqa: E731
            if r.model == "baseline":
                u.append(Uitkomst("twijfel", "Het gekozen model is de baseline: geen enkel model voegt dan iets toe.", ("metrics",)))
            elif base and winst(r.gemiddelde, base.gemiddelde) < r.sd:
                u.append(Uitkomst("twijfel", f"De winst ten opzichte van de baseline ({winst(r.gemiddelde, base.gemiddelde):.3f}) is "
                                             f"kleiner dan de spreiding over de folds ({r.sd:.3f}).", ("cawley",)))
            if beste and beste.naam != n and winst(beste.gemiddelde, r.gemiddelde) > r.sd:
                u.append(Uitkomst("twijfel", f"'{beste.naam}' scoort duidelijk beter ({beste.gemiddelde:.3f} tegen {r.gemiddelde:.3f}). "
                                             "Onderbouw waarom je toch dit model kiest (bijv. eenvoud of uitlegbaarheid).", ("cawley",)))
            kloof = winst(r.train, r.gemiddelde)
            relatief = kloof / abs(r.gemiddelde) if lab.cv.metriek in ("rmse", "mae") and r.gemiddelde else kloof
            if relatief > (.3 if lab.cv.metriek in ("rmse", "mae") else .15):
                u.append(Uitkomst("twijfel", f"Groot verschil tussen train ({r.train:.3f}) en validatie ({r.gemiddelde:.3f}): "
                                             "het model overfit waarschijnlijk. Overweeg meer regularisatie of een eenvoudiger model.", ("esl",)))
    return u
