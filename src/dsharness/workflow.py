"""Workflow F1 → F2 → SPLIT → F3 → F4 → F5. Gewone Python: de volgorde en de gates liggen vast, geen LLM beslist dat."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic_ai import Agent

from .acties import VASTE_REGELS, omschrijf
from .data import DataLaag, KluisFout
from .lab import Lab
from .modeler import Modeler, ModelerDeps
from .po import PO
from .prep import PrepDeps, PrepEngineer
from .prep_plan import PrepPlan, PrepPlanFout
from .rapport import ConceptVoorstel, Verkenningsrapport
from .reviewer import ReviewContext, Reviewer
from .scout import Scout, ScoutDeps
from .tools import AgentDeps
from .verkenner import F3Rapport


@dataclass
class Team:
    scout: Scout
    reviewer: Reviewer
    verkenner: Agent[AgentDeps, F3Rapport] | None = None
    prep: PrepEngineer | None = None
    modeler: Modeler | None = None
    max_reviewer_revisies: int = 1  # bij 'fout' mag de specialist eerst zelf herzien
    max_po_rondes: int = 3
    max_prep_rondes: int = 3  # aanvullende rondes tot de pipeline model-klaar is
    max_model_rondes: int = 4  # rondes experimenten tot er een model gekozen is


@dataclass
class Uitkomst:
    rapport: Verkenningsrapport
    uitgevoerd: list[str]
    gesplitst: bool


def _vragen(po_vragen):
    return po_vragen if po_vragen is not None else []


async def eerste_blik_tot_split(po: PO, data: DataLaag, *, doel: str, taak: str, team: Team,
                                po_vragen: list[tuple[str, str]] | None = None) -> Uitkomst:
    deps = ScoutDeps(data=data, po=po, doel=doel, taak=taak, po_vragen=_vragen(po_vragen))
    po.melding("Data Scout verkent de data (F1, vóór de split)…")
    rapport = (await team.scout.verken.run("Verken de data en lever je rapport.", deps=deps)).output
    po.toon_rapport(rapport)

    ctx = ReviewContext(data=data, taak=taak, bevindingen=rapport.bevindingen, po_vragen=deps.po_vragen)
    uitgevoerd: list[str] = []
    nummers = {"F2": 0, "SPLIT": 0}
    for concept in rapport.voorstellen:
        fase = "F2" if concept.actie.type in VASTE_REGELS else "SPLIT"
        nummers[fase] += 1
        resultaat = await _behandel(po, concept, f"{fase}-{nummers[fase]:03d}", fase, ctx, deps, team,
                                    herzie=team.scout.herzie, uitvoer=data.voer_uit)
        if resultaat:
            uitgevoerd.append(resultaat)
        elif fase == "SPLIT":
            po.melding("Geen split uitgevoerd. F3 (verkennen op train) kan pas starten na een split.")
    return Uitkomst(rapport, uitgevoerd, data.gesplitst)


async def verken_train(po: PO, data: DataLaag, *, doel: str, taak: str, team: Team,
                       po_vragen: list[tuple[str, str]] | None = None) -> F3Rapport:
    """F3: verkennen op de trainset. Verandert niets aan de data; de PO beantwoordt alleen vragen."""
    if not data.gesplitst:
        raise KluisFout("F3 kan pas na de split.")
    if team.verkenner is None:
        raise ValueError("Geen Verkenner in het team.")
    deps = AgentDeps(data=data, po=po, doel=doel, taak=taak, po_vragen=_vragen(po_vragen))
    deps.max_vragen = len(deps.po_vragen) + int(data.zicht().isna().any().sum()) + 3
    po.melding("Verkenner onderzoekt de trainset (F3)…")
    rapport = (await team.verkenner.run("Verken de trainset en lever je F3-rapport.", deps=deps)).output
    po.toon_rapport(rapport)
    return rapport


async def prepareer(po: PO, data: DataLaag, *, doel: str, taak: str, team: Team, f3: F3Rapport | None = None,
                    po_vragen: list[tuple[str, str]] | None = None, plan: PrepPlan | None = None) -> PrepPlan:
    """F4: de Prep Engineer stelt stappen voor; goedgekeurde stappen komen in het plan, tot het model-klaar is."""
    if not data.gesplitst:
        raise KluisFout("F4 kan pas na de split.")
    if team.prep is None:
        raise ValueError("Geen Prep Engineer in het team.")
    plan = plan or PrepPlan(data.target)
    df = data.zicht()
    deps = PrepDeps(data=data, po=po, doel=doel, taak=taak, po_vragen=_vragen(po_vragen), plan=plan,
                    profielen=f3.profielen if f3 else [], voor_f4=f3.voor_f4 if f3 else [])
    deps.max_vragen = len(deps.po_vragen) + 4
    ctx = ReviewContext(data=data, taak=taak, po_vragen=deps.po_vragen,
                        profielen={p.kolom: p for p in f3.profielen} if f3 else {}, plan=plan)
    po.melding("Prep Engineer stelt de preparatie voor (F4)…")
    prompt, n = "Stel de preparatiestappen voor.", 0
    for _ in range(team.max_prep_rondes):
        voorstellen = (await team.prep.voorstel.run(prompt, deps=deps)).output.voorstellen
        for c in voorstellen:
            n += 1
            await _behandel(po, c, f"F4-{n:03d}", "F4 Preparatie", ctx, deps, team,
                            herzie=team.prep.herzie, uitvoer=lambda a: plan.voeg_toe(a, df))
        if not (open_ := plan.problemen(df)):
            break
        po.melding("De pipeline is nog niet model-klaar: " + "; ".join(open_) + ". De Prep Engineer vult aan.")
        prompt = ("Niet alle stappen zijn goedgekeurd, of er ontbreekt nog iets. Stel aanvullende stappen voor voor: "
                  + "; ".join(open_) + ". Houd rekening met de besluiten en motivaties van de PO.")
    else:
        po.melding("F4 gestopt: de pipeline is nog niet model-klaar. " + "; ".join(plan.problemen(df)))
        return plan
    try:
        proef = plan.proef(df)
    except PrepPlanFout as e:
        po.melding(f"De pipeline werkt nog niet op train: {e}")
        return plan
    po.toon_plan(plan, proef)
    return plan


async def modelleer(po: PO, data: DataLaag, *, doel: str, taak: str, team: Team, plan: PrepPlan,
                    po_vragen: list[tuple[str, str]] | None = None, positief: str | None = None, lab: Lab | None = None) -> Lab:
    """F5: rondes van voorstellen (CV-opzet, experimenten, modelkeuze) tot er een model gekozen is."""
    if team.modeler is None:
        raise ValueError("Geen Modeler in het team.")
    if plan.problemen(data.zicht()):
        raise ValueError("De F4-pipeline is nog niet model-klaar.")
    lab = lab or Lab(data, plan, taak, positief)
    deps = ModelerDeps(data=data, po=po, doel=doel, taak=taak, po_vragen=_vragen(po_vragen), lab=lab)
    deps.max_vragen = len(deps.po_vragen) + 3
    ctx = ReviewContext(data=data, taak=taak, po_vragen=deps.po_vragen, plan=plan, lab=lab)
    po.melding("Modeler gaat aan de slag (F5, cross-validatie op train)…")
    n = 0
    for ronde in range(1, team.max_model_rondes + 1):
        prompt = ("Eerste ronde: vraag de PO wat zwaar weegt, leg de CV-opzet vast en stel een baseline en enkele modellen voor."
                  if lab.cv is None else f"Ronde {ronde}: bekijk de resultaten. Stel gerichte experimenten voor, of kies een model.")
        for c in (await team.modeler.voorstel.run(prompt, deps=deps)).output.voorstellen:
            n += 1
            await _behandel(po, c, f"F5-{n:03d}", "F5 Modelleren", ctx, deps, team, herzie=team.modeler.herzie, uitvoer=lab.voer_uit)
            if lab.gekozen:
                break
        if lab.resultaten:
            po.toon_resultaten(lab)
        if lab.gekozen:
            return lab
    po.melding(f"F5 gestopt na {team.max_model_rondes} rondes zonder gekozen model.")
    return lab


async def _behandel(po: PO, c: ConceptVoorstel, id: str, fase: str, ctx: ReviewContext, deps, team: Team, *,
                    herzie: Agent, uitvoer: Callable) -> str | None:
    """Review → (herzien) → PO → uitvoeren. Geeft de uitvoermelding, of None als er niets is uitgevoerd."""
    reviewer_revisies = po_rondes = 0
    while True:
        review = await team.reviewer.beoordeel(c, ctx)
        if review.oordeel == "fout" and reviewer_revisies < team.max_reviewer_revisies:
            reviewer_revisies += 1
            po.melding(f"Reviewer vindt voorstel {id} fout; de specialist herziet het.")
            c = await _herzie(herzie, deps, c, "de Reviewer", review.toelichting)
            continue
        besluit = await po.vraag_akkoord(c.naar_voorstel(id, fase, review))
        if besluit.keuze in ("akkoord", "overrulen"):
            try:
                resultaat = uitvoer(c.actie)
            except (ValueError, KluisFout) as e:  # bijv. een overrulede stap die niet kan
                po.melding(f"Voorstel {id} kon niet worden uitgevoerd: {e}")
                return None
            po.melding(resultaat)
            return resultaat
        if besluit.keuze == "afwijzen":
            po.melding(f"Voorstel {id} afgewezen: {omschrijf(c.actie)}.")
            return None
        po_rondes += 1  # aanpassen
        if po_rondes >= team.max_po_rondes:
            po.melding(f"Voorstel {id}: na {po_rondes} aanpassingen gestopt. Niet uitgevoerd.")
            return None
        reviewer_revisies = 0
        c = await _herzie(herzie, deps, c, "de PO", besluit.motivatie)


async def _herzie(agent: Agent, deps, c: ConceptVoorstel, door: str, feedback: str) -> ConceptVoorstel:
    prompt = (f"Herzie dit voorstel op basis van de feedback van {door}. Verken opnieuw als dat nodig is.\n"
              f"Voorstel:\n{c.model_dump_json(indent=1)}\n\nFeedback van {door}: {feedback}")
    return (await agent.run(prompt, deps=deps)).output
