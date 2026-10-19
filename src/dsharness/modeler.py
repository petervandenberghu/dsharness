"""Modeler: fase F5. Stelt een CV-opzet, experimenten en een modelkeuze voor; het Lab voert ze uit op train."""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext

from .acties import F5_TYPES
from .bronnen import catalogus_tekst
from .lab import CATALOGUS, Lab
from .modellen import maak_model
from .rapport import ConceptVoorstel
from .tools import AgentDeps, basis_tools, train_tools


@dataclass
class ModelerDeps(AgentDeps):
    lab: Lab | None = None


class ModelerVoorstellen(BaseModel):
    voorstellen: list[ConceptVoorstel] = Field(description="F5-acties in uitvoervolgorde")


PARAMS = "\n".join(f"- {m}: {', '.join(sorted(p)) or 'geen parameters'}" for m, (_, _, p) in CATALOGUS.items())

INSTRUCTIES = f"""Je bent de Modeler in een data science-team. Je werkt in fase F5: modelleren met cross-validatie op de trainset.
De testset zit in een kluis en komt pas in F6, één keer. De gebruiker is de product owner (PO). Elke actie keurt de PO goed.
De preparatie-pipeline uit F4 zit automatisch vóór elk model en wordt binnen elke fold opnieuw gefit.

Werkwijze, in rondes:
1. Eerste ronde: vraag de PO (vraag_po) welke fouten het zwaarst wegen, en leg dan de cv_opzet vast:
   - methode: groep als de split per groep was, tijd als de split op tijd was, stratified bij een klasse-target, anders kfold.
   - metriek: regressie rmse/mae/r2; classificatie accuracy alleen bij evenwichtige klassen, anders recall, f1,
     balanced_accuracy of roc_auc, afhankelijk van wat de PO belangrijk vindt.
   - folds: meestal 5; minder als er weinig rijen of groepen zijn.
   Stel daarna een baseline voor (model 'baseline') en 2 à 3 verschillende modellen.
2. Volgende rondes: bekijk de resultaten. Let op het verschil tussen train en validatie (overfitting) en de spreiding.
   Stel gerichte experimenten voor (bijv. meer regularisatie, andere parameters, feature-selectie, balanceren),
   of kies een model met kies_model als verder zoeken weinig oplevert. Hou het aantal experimenten beperkt.
3. Kies niet automatisch de hoogste score: is het verschil kleiner dan de spreiding, dan wint het eenvoudigere model.
Modellen en toegestane parameters (alleen deze):
{PARAMS}
(alpha alleen bij regressie, C alleen bij classificatie; gebalanceerd alleen bij classificatie.)

Vul per voorstel: observatie (met getallen), onderzocht, po_kennis, plan, waarom, en bron_sleutels uit deze lijst:
{catalogus_tekst()}
Schrijf kort en in eenvoudig Nederlands."""


def opdracht(ctx: RunContext[ModelerDeps]) -> str:
    d = ctx.deps
    split = d.data.split_actie
    eerder = "\n".join(f"- {v} → {a}" for v, a in d.po_vragen) or "geen"
    return (f"Projectdoel: {d.doel}\nTarget: '{d.data.target}' ({d.taak}). Trainset: {d.data.n_train} rijen. "
            f"Split: {split.methode if split else '-'}"
            f"{f' (groep: {split.groep_kolom})' if split and split.groep_kolom else ''}"
            f"{f' (tijd: {split.tijd_kolom})' if split and split.tijd_kolom else ''}.\n"
            f"Features na de preparatie: {d.lab.n_features() if d.lab else '?'}.\n"
            f"Stand van het lab:\n{d.lab.overzicht() if d.lab else '-'}\nEerder aan de PO gevraagd:\n{eerder}")


def controleer(lab: Lab, voorstellen: list[ConceptVoorstel]) -> list[str]:
    """Statische controle van de volgorde, zonder iets te draaien."""
    fouten = []
    cv = lab.cv is not None
    baseline = lab.baseline() is not None
    namen = set(lab.experimenten)
    for c in voorstellen:
        a = c.actie
        if a.type not in F5_TYPES:
            fouten.append(f"'{a.type}' is geen F5-actie")
        elif a.type == "cv_opzet":
            if cv or lab.resultaten:
                fouten.append("de CV-opzet ligt al vast")
            cv = True
        elif a.type == "experiment":
            if not cv:
                fouten.append(f"'{a.naam}': leg eerst de cv_opzet vast")
            if a.naam in namen:
                fouten.append(f"de naam '{a.naam}' is al gebruikt")
            if a.model != "baseline" and not baseline:
                fouten.append(f"'{a.naam}': stel eerst een baseline voor")
            if fout := sorted(set(a.parameters) - CATALOGUS[a.model][2]):
                fouten.append(f"'{a.naam}': onbekende parameters {fout}")
            baseline = baseline or a.model == "baseline"
            namen.add(a.naam)
        elif a.type == "kies_model" and a.experiment not in lab.resultaten:
            fouten.append(f"kies_model kan alleen een experiment dat al resultaten heeft (nu: {sorted(lab.resultaten) or 'geen'})")
    if not voorstellen:
        fouten.append("stel minstens één actie voor")
    return fouten


@dataclass
class Modeler:
    voorstel: Agent[ModelerDeps, ModelerVoorstellen]
    herzie: Agent[ModelerDeps, ConceptVoorstel]


def maak_modeler(model=None) -> Modeler:
    m, tools = maak_model(model), [basis_tools(), train_tools()]
    voorstel = Agent(m, deps_type=ModelerDeps, output_type=ModelerVoorstellen, instructions=[INSTRUCTIES, opdracht],
                     toolsets=tools, name="modeler", retries=4)
    herzie = Agent(m, deps_type=ModelerDeps, output_type=ConceptVoorstel, instructions=[INSTRUCTIES, opdracht],
                   toolsets=tools, name="modeler_herzien", retries=3)

    @voorstel.output_validator
    def _volgorde(ctx: RunContext[ModelerDeps], uit: ModelerVoorstellen) -> ModelerVoorstellen:
        if fouten := controleer(ctx.deps.lab, uit.voorstellen):
            raise ModelRetry("Deze voorstellen kloppen niet: " + "; ".join(fouten) + ".")
        return uit

    @herzie.output_validator
    def _f5(ctx: RunContext[ModelerDeps], uit: ConceptVoorstel) -> ConceptVoorstel:
        if uit.actie.type not in F5_TYPES:
            raise ModelRetry(f"Stel een F5-actie voor ({', '.join(F5_TYPES)}), niet '{uit.actie.type}'.")
        return uit

    return Modeler(voorstel=voorstel, herzie=herzie)
