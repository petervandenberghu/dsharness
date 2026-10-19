"""Reviewer: vaste regels in code, plus een optionele LLM-laag.

De LLM mag een oordeel strenger maken, nooit milder: het eindoordeel is het strengste van de twee.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .acties import OnmogelijkNaarNA, Split, VerwijderDubbelen, VerwijderZonderTarget
from .bronnen import BronSleutel, bron, catalogus_tekst
from .contracts import ReviewOordeel
from .data import DataLaag
from .modellen import maak_model
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
    return u or [Uitkomst("goed", "Geen bezwaren gevonden in de regels.")]


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
