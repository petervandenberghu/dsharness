"""Workflow F1 → F2 → SPLIT → F3. Gewone Python: de volgorde en de gates liggen vast, geen LLM beslist dat."""
from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent

from .acties import VASTE_REGELS
from .data import DataLaag, KluisFout
from .po import PO
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
    max_reviewer_revisies: int = 1  # bij 'fout' mag de specialist eerst zelf herzien
    max_po_rondes: int = 3


@dataclass
class Uitkomst:
    rapport: Verkenningsrapport
    uitgevoerd: list[str]
    gesplitst: bool


async def eerste_blik_tot_split(po: PO, data: DataLaag, *, doel: str, taak: str, team: Team,
                                po_vragen: list[tuple[str, str]] | None = None) -> Uitkomst:
    deps = ScoutDeps(data=data, po=po, doel=doel, taak=taak, po_vragen=po_vragen if po_vragen is not None else [])
    po.melding("Data Scout verkent de data (F1, vóór de split)…")
    rapport = (await team.scout.verken.run("Verken de data en lever je rapport.", deps=deps)).output
    po.toon_rapport(rapport)

    ctx = ReviewContext(data=data, taak=taak, bevindingen=rapport.bevindingen, po_vragen=deps.po_vragen)
    uitgevoerd: list[str] = []
    nummers = {"F2": 0, "SPLIT": 0}
    for concept in rapport.voorstellen:
        fase = "F2" if concept.actie.type in VASTE_REGELS else "SPLIT"
        nummers[fase] += 1
        resultaat = await _behandel(po, concept, f"{fase}-{nummers[fase]:03d}", fase, ctx, deps, team)
        if resultaat:
            uitgevoerd.append(resultaat)
            po.melding(resultaat)
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
    deps = AgentDeps(data=data, po=po, doel=doel, taak=taak, po_vragen=po_vragen if po_vragen is not None else [])
    n_na = int(data.zicht().isna().any().sum())
    deps.max_vragen = len(deps.po_vragen) + n_na + 3
    po.melding("Verkenner onderzoekt de trainset (F3)…")
    rapport = (await team.verkenner.run("Verken de trainset en lever je F3-rapport.", deps=deps)).output
    po.toon_rapport(rapport)
    return rapport


async def _behandel(po: PO, c: ConceptVoorstel, id: str, fase: str, ctx: ReviewContext, deps: ScoutDeps, team: Team) -> str | None:
    """Review → (herzien) → PO → uitvoeren. Geeft de uitvoermelding, of None als er niets is uitgevoerd."""
    reviewer_revisies = po_rondes = 0
    while True:
        review = await team.reviewer.beoordeel(c, ctx)
        if review.oordeel == "fout" and reviewer_revisies < team.max_reviewer_revisies:
            reviewer_revisies += 1
            po.melding(f"Reviewer vindt voorstel {id} fout; de Data Scout herziet het.")
            c = await _herzie(team.scout, deps, c, "de Reviewer", review.toelichting)
            continue
        besluit = await po.vraag_akkoord(c.naar_voorstel(id, fase, review))
        if besluit.keuze in ("akkoord", "overrulen"):
            return ctx.data.voer_uit(c.actie)
        if besluit.keuze == "afwijzen":
            po.melding(f"Voorstel {id} afgewezen.")
            return None
        po_rondes += 1  # aanpassen
        if po_rondes >= team.max_po_rondes:
            po.melding(f"Voorstel {id}: na {po_rondes} aanpassingen gestopt. Niet uitgevoerd.")
            return None
        reviewer_revisies = 0
        c = await _herzie(team.scout, deps, c, "de PO", besluit.motivatie)


async def _herzie(scout: Scout, deps: ScoutDeps, c: ConceptVoorstel, door: str, feedback: str) -> ConceptVoorstel:
    prompt = (f"Herzie dit voorstel op basis van de feedback van {door}. Verken opnieuw als dat nodig is.\n"
              f"Voorstel:\n{c.model_dump_json(indent=1)}\n\nFeedback van {door}: {feedback}")
    return (await scout.herzie.run(prompt, deps=deps)).output
