"""Verkenner: fase F3, verkennen op de trainset. Levert per kolom een profiel als input voor F4.

F3 verandert niets aan de data, dus er is geen akkoord nodig. De Verkenner vraagt de PO wel om domeinkennis.
Controles in code: elke kolom een profiel, NA-mechanisme alleen na NA-patroononderzoek én een vraag aan de PO.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext

from .data import DataLaag
from .modellen import maak_model
from .tools import AgentDeps, basis_tools, train_tools

NAMechanisme = Literal["geen_na", "MCAR", "MAR", "MNAR", "onbekend"]


class KolomProfiel(BaseModel):
    kolom: str
    type: Literal["getal", "categorie"]
    na_aantal: int = Field(ge=0)
    na_mechanisme: NAMechanisme
    na_onderbouwing: str = Field("", description="Uitkomst van na_patroon én wat de PO zei. Leeg als er geen NA is.")
    verdeling: str = Field(description="Bijv. 'rechtsscheef (1,4), mediaan 107' of '4 categorieën, 1 zeldzaam'")
    uitschieters: str = Field("", description="Aantal, en of het fouten of echte waarden zijn (volgens data/PO)")
    relatie_target: str = Field("", description="Bijv. 'r = 0,75' of 'Maand: 26% opgezegd, Tweejaar: 0%'")
    aandachtspunten: list[str] = []


class F3Rapport(BaseModel):
    samenvatting: str
    target: str = Field(description="Verdeling of klassenbalans van de target in train")
    profielen: list[KolomProfiel] = Field(description="Eén profiel per kolom behalve de target")
    voor_f4: list[str] = Field(description="Concrete aandachtspunten voor de preparatie, met de kolom erbij")


INSTRUCTIES = """Je bent de Verkenner in een data science-team. Je werkt in fase F3: verkennen op de trainset, ná de split.
De testset zit in een kluis; je tools zien alleen train. De gebruiker is de product owner (PO): alleen de PO weet iets
over de business en de herkomst van de data.

Werkwijze:
1. Begin met na_telling en correlatie. Bekijk per kolom de verdeling en, waar het zinvol is, uitschieters en target_relatie.
2. Voor elke kolom met NA: draai na_patroon én vraag de PO waarom de waarden ontbreken (noem de kolom en de getallen).
   Bepaal dan het mechanisme:
   - MCAR: het ontbreken is toeval (geen patroon, of te weinig NA's, en de PO weet van geen reden).
   - MAR: het ontbreken hangt samen met andere kolommen die je wél hebt.
   - MNAR: het ontbreken hangt samen met de ontbrekende waarde zelf. Dat kun je niet uit de data bewijzen; alleen de PO
     kan het bevestigen. Gaat de PO-kennis tegen het datapatroon in, volg dan de PO en leg uit waarom.
   - onbekend: als data en PO samen geen conclusie toelaten.
3. Bij uitschieters: vraag de PO of het fouten of echte waarden zijn, als je dat niet uit de data kunt halen.
4. Je verandert niets aan de data en stelt nog geen acties voor. Je levert kennis voor de volgende fase.

Lever een F3-rapport: per kolom (behalve de target) een profiel, een korte beschrijving van de target, en concrete
aandachtspunten voor de preparatie (bijv. 'bouwjaar: MAR, 16% NA → imputatie die andere kolommen gebruikt').
Schrijf kort en in eenvoudig Nederlands. Gebruik getallen."""


def opdracht(ctx: RunContext[AgentDeps]) -> str:
    d = ctx.deps
    eerder = "\n".join(f"- {v} → {a}" for v, a in d.po_vragen) or "geen"
    gedaan = "\n".join(f"- {g}" for g in d.data.geschiedenis) or "niets"
    return (f"Projectdoel: {d.doel}\nTarget: '{d.data.target}' ({d.taak}). Trainset: {d.data.n_train} rijen.\n"
            f"Al uitgevoerd (stel dit niet opnieuw voor):\n{gedaan}\n"
            f"Eerder aan de PO gevraagd (hoef je niet opnieuw te vragen):\n{eerder}")


def controle(r: F3Rapport, data: DataLaag, deps: AgentDeps) -> list[str]:
    """Wat er aan het rapport mankeert, vergeleken met de data en met wat er echt onderzocht en gevraagd is."""
    df = data.zicht()
    kolommen = [k for k in df.columns if k != data.target]
    profielen = {p.kolom: p for p in r.profielen}
    mist: list[str] = []
    if onbekend := sorted(set(profielen) - set(kolommen)):
        mist.append(f"profielen voor kolommen die niet bestaan of de target zijn: {onbekend}")
    for k in kolommen:
        p = profielen.get(k)
        if p is None:
            mist.append(f"geen profiel voor '{k}'")
            continue
        n_na = int(df[k].isna().sum())
        soort = "getal" if pd.api.types.is_numeric_dtype(df[k]) else "categorie"
        if p.type != soort:
            mist.append(f"'{k}' is een {soort}, niet een {p.type}")
        if p.na_aantal != n_na:
            mist.append(f"'{k}' heeft {n_na} NA in train, niet {p.na_aantal}")
        if n_na == 0:
            if p.na_mechanisme != "geen_na":
                mist.append(f"'{k}' heeft geen NA; gebruik na_mechanisme 'geen_na'")
            continue
        if p.na_mechanisme == "geen_na":
            mist.append(f"'{k}' heeft {n_na} NA; bepaal het mechanisme")
        if not any(v.tool == "na_patroon" and v.kolom == k and v.gesplitst for v in data.log):
            mist.append(f"'{k}': draai eerst na_patroon")
        if not deps.po_gevraagd_over(k):
            mist.append(f"'{k}': vraag de PO waarom waarden ontbreken (noem de kolomnaam)")
    return mist


def maak_verkenner(model=None) -> Agent[AgentDeps, F3Rapport]:
    verkenner = Agent(maak_model(model), deps_type=AgentDeps, output_type=F3Rapport, instructions=[INSTRUCTIES, opdracht],
                      toolsets=[basis_tools(), train_tools()], name="verkenner", retries=4)

    @verkenner.output_validator
    def _controle(ctx: RunContext[AgentDeps], uit: F3Rapport) -> F3Rapport:
        if mist := controle(uit, ctx.deps.data, ctx.deps):
            raise ModelRetry("Je F3-rapport klopt nog niet: " + "; ".join(mist) + ".")
        return uit

    return verkenner
