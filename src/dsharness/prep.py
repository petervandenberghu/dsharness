"""Prep Engineer: fase F4. Stelt preparatiestappen voor; goedgekeurde stappen gaan in het PrepPlan (sklearn-Pipeline)."""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext

from .acties import F4_TYPES, omschrijf
from .bronnen import catalogus_tekst
from .modellen import maak_model
from .prep_plan import PrepPlan, PrepPlanFout
from .rapport import ConceptVoorstel
from .tools import PO_REGELS, AgentDeps, basis_tools, train_tools


@dataclass
class PrepDeps(AgentDeps):
    plan: PrepPlan | None = None
    profielen: list = field(default_factory=list)  # KolomProfiel's uit F3
    voor_f4: list[str] = field(default_factory=list)


class PrepVoorstellen(BaseModel):
    voorstellen: list[ConceptVoorstel] = Field(description="F4-stappen in logische volgorde")


INSTRUCTIES = f"""Je bent de Prep Engineer in een data science-team. Je werkt in fase F4: de preparatie, ná de split.
De gebruiker is de product owner (PO). Je stelt stappen voor; de Reviewer en de PO keuren elke stap goed.
Goedgekeurde stappen komen in een sklearn-Pipeline die alleen op de trainset wordt gefit. De volgorde per kolom
(afkappen → log → imputeren → schalen → encoderen) legt de pipeline zelf vast.

Gebruik de kolomprofielen uit F3. Richtlijnen (Van Buuren 2018; Kuhn & Johnson 2019; scikit-learn):
- Kolommen die geen feature zijn (ID's, of meer dan ~50% NA en weinig nut): verwijder_kolom.
- NA, MCAR, weinig NA, symmetrisch: gemiddelde. Scheef of uitschieters: mediaan. Categorie: modus of constante 'onbekend'.
- NA, MAR of veel NA: knn of iteratief (gebruikt andere kolommen). Kies één van de twee voor de hele pipeline.
- NA, MNAR: eerst een na_indicator, dan imputeren.
- Uitschieters: fouten waren al NA gemaakt in F2. Echte waarden: houden; eventueel log (sterk rechtsscheef) of afkappen.
- Categorieën: one-hot bij weinig categorieën, target encoding bij veel.
- Schalen: nodig voor lineaire modellen met regularisatie, KNN, SVM en neurale netten. Eén schaal-stap, meestal 'alle'.
Elke numerieke kolom met NA heeft een imputatie nodig, en elke categorische kolom een encoding; anders kan geen model
ermee werken. Stel geen stappen voor die niets doen. Vraag de PO als je domeinkennis mist (bijv. zijn uitschieters echt?).

Vul per voorstel: observatie (met getallen uit F3 of je tools), onderzocht, po_kennis (wat de PO zei, of 'niet gevraagd'),
plan, waarom, en bron_sleutels uit deze lijst:
{catalogus_tekst()}
Schrijf kort en in eenvoudig Nederlands."""


def opdracht(ctx: RunContext[PrepDeps]) -> str:
    d = ctx.deps
    profielen = "\n".join(
        f"- {p.kolom} ({p.type}): NA {p.na_aantal} [{p.na_mechanisme}] {p.na_onderbouwing} | verdeling: {p.verdeling}"
        f" | uitschieters: {p.uitschieters or '-'} | target: {p.relatie_target or '-'}" for p in d.profielen) or "geen F3-profielen"
    plan = "\n".join(f"- {omschrijf(s)}" for s in d.plan.stappen) if d.plan and d.plan.stappen else "nog leeg"
    eerder = "\n".join(f"- {v} → {a}" for v, a in d.po_vragen) or "geen"
    return (f"Projectdoel: {d.doel}\nTarget: '{d.data.target}' ({d.taak}). Trainset: {d.data.n_train} rijen.\n"
            f"Kolomprofielen uit F3:\n{profielen}\nAandachtspunten uit F3: {d.voor_f4 or '-'}\n"
            f"Al in het plan (niet opnieuw voorstellen):\n{plan}\nEerder aan de PO gevraagd:\n{eerder}")


def simuleer(plan: PrepPlan, voorstellen: list[ConceptVoorstel], df) -> list[str]:
    """Voeg de voorstellen toe aan een kopie van het plan; geef terug wat er mis is."""
    kopie = PrepPlan(plan.target)
    kopie.stappen = list(plan.stappen)
    fouten = []
    for c in voorstellen:
        if c.actie.type not in F4_TYPES:
            fouten.append(f"'{c.actie.type}' is geen F4-stap")
            continue
        try:
            kopie.voeg_toe(c.actie, df)
        except PrepPlanFout as e:
            fouten.append(f"{omschrijf(c.actie)}: {e}")
    return fouten + kopie.problemen(df)


@dataclass
class PrepEngineer:
    voorstel: Agent[PrepDeps, PrepVoorstellen]
    herzie: Agent[PrepDeps, ConceptVoorstel]


def maak_prep(model=None) -> PrepEngineer:
    m, tools = maak_model(model), [basis_tools(), train_tools()]
    voorstel = Agent(m, deps_type=PrepDeps, output_type=PrepVoorstellen, instructions=[INSTRUCTIES, PO_REGELS, opdracht],
                     toolsets=tools, name="prep_engineer", retries=4)
    herzie = Agent(m, deps_type=PrepDeps, output_type=ConceptVoorstel, instructions=[INSTRUCTIES, PO_REGELS, opdracht],
                   toolsets=tools, name="prep_engineer_herzien", retries=3)

    @voorstel.output_validator
    def _compleet(ctx: RunContext[PrepDeps], uit: PrepVoorstellen) -> PrepVoorstellen:
        if fouten := simuleer(ctx.deps.plan, uit.voorstellen, ctx.deps.data.zicht()):
            raise ModelRetry("Met deze voorstellen klopt de pipeline nog niet: " + "; ".join(fouten) + ".")
        return uit

    @herzie.output_validator
    def _f4(ctx: RunContext[PrepDeps], uit: ConceptVoorstel) -> ConceptVoorstel:
        if uit.actie.type not in F4_TYPES:
            raise ModelRetry(f"Stel een F4-stap voor ({', '.join(F4_TYPES)}), niet '{uit.actie.type}'.")
        return uit

    return PrepEngineer(voorstel=voorstel, herzie=herzie)
