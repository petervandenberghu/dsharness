"""Tools voor de agents, als toolsets die meerdere agents kunnen delen.

Elke tool leest via DataLaag.zicht() en wordt genoteerd in data.log, zodat de harness kan controleren wat echt onderzocht is.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset

from . import verkennen as vk
from .data import DataLaag
from .po import PO


@dataclass
class AgentDeps:
    data: DataLaag
    po: PO
    doel: str
    taak: str
    po_vragen: list[tuple[str, str]] = field(default_factory=list)
    max_vragen: int = 6

    def po_gevraagd_over(self, kolom: str, *woorden: str) -> bool:
        """Is de PO gevraagd over deze kolom (en, als gegeven, over een van deze onderwerpen)?"""
        for v, _ in self.po_vragen:
            v = v.lower()
            if kolom.lower() in v and (not woorden or any(w in v for w in woorden)):
                return True
        return False


PO_REGELS = """Vragen aan de PO (tool vraag_po):
- Vraag alleen wat je níet uit de data kunt halen: de betekenis van een kolom of waarde, de herkomst, waarom iets
  ontbreekt, of een opvallende of extreme waarde een fout of echt is, en wat zakelijk belangrijk is.
- Statistische eigenschappen (verdeling, scheefheid, correlatie, aantallen) bepaal je zelf met je tools; vraag die niet.
- Elke deelvraag bevat één vraag (één vraagteken). Meerdere vragen tegelijk mag, als lijst: de PO beantwoordt ze los.
- Vul per deelvraag de kolom in en geef als context de getallen die je zag."""

# Vragen die over de data zelf gaan en die de onderzoeker dus zelf moet beantwoorden.
DATAVRAAG = re.compile(r"scheef|verdeling|verdeeld|correlatie|correleert|standaardafwijking|histogram|kwartiel|mediaan"
                       r"|hoeveel (rijen|waarden|na)|normaal ?verdeel", re.IGNORECASE)


class Deelvraag(BaseModel):
    kolom: str | None = Field(None, description="De kolom waar de vraag over gaat; leeg bij een algemene vraag")
    vraag: str = Field(description="Eén vraag, met één vraagteken")
    context: str = Field("", description="De getallen die je zag, zodat de PO weet waar het over gaat")

    def tekst(self) -> str:
        if self.kolom and self.kolom.lower() not in self.vraag.lower():
            return f"[{self.kolom}] {self.vraag}"
        return self.vraag


def controleer_vragen(vragen: list[Deelvraag], kolommen: list[str]) -> list[str]:
    fouten = []
    if not vragen:
        fouten.append("stel minstens één vraag")
    for i, q in enumerate(vragen, 1):
        if q.vraag.count("?") > 1:
            fouten.append(f"deelvraag {i} bevat meerdere vragen; maak er losse deelvragen van, zodat de PO ze apart kan beantwoorden")
        if DATAVRAAG.search(q.vraag):
            fouten.append(f"deelvraag {i} ('{q.vraag[:70]}') gaat over iets wat je zelf in de data kunt zien; gebruik je tools")
        if q.kolom and q.kolom not in kolommen:
            fouten.append(f"deelvraag {i}: kolom '{q.kolom}' bestaat niet (kies uit {kolommen})")
    return fouten


def _zicht(ctx: RunContext[AgentDeps], tool: str, kolom: str | None = None, *, alleen_train: bool = False):
    d = ctx.deps.data
    if alleen_train and not d.gesplitst:
        raise ModelRetry(f"'{tool}' mag pas na de split: vóór de split kijk je anders mee in de testset.")
    if kolom is not None and kolom not in (kolommen := list(d.zicht().columns)):
        raise ModelRetry(f"Kolom '{kolom}' bestaat niet. Kies uit: {kolommen}")
    d.noteer(tool, kolom)
    return d.zicht()


def basis_tools() -> FunctionToolset[AgentDeps]:
    """Controleren en structuur: mag vóór en na de split."""
    ts: FunctionToolset[AgentDeps] = FunctionToolset()

    @ts.tool
    def vorm(ctx: RunContext[AgentDeps]) -> dict:
        """Aantal rijen, en per kolom het type, het aantal verschillende waarden en 3 voorbeelden."""
        return vk.vorm(_zicht(ctx, "vorm"))

    @ts.tool
    def dubbelen(ctx: RunContext[AgentDeps]) -> dict:
        """Rijen die in alle kolommen gelijk zijn aan een andere rij."""
        return vk.dubbelen(_zicht(ctx, "dubbelen"))

    @ts.tool
    def na_telling(ctx: RunContext[AgentDeps]) -> dict:
        """Aantal en fractie ontbrekende waarden per kolom."""
        return vk.na_telling(_zicht(ctx, "na_telling"))

    @ts.tool
    def beschrijf(ctx: RunContext[AgentDeps]) -> dict:
        """Per numerieke kolom: min, kwartielen, gemiddelde, max, scheefheid, aantal nullen en negatieve waarden.
        Per categorische kolom: aantal verschillende waarden en de meest voorkomende."""
        return vk.beschrijf(_zicht(ctx, "beschrijf"))

    @ts.tool
    def tel_waarden(ctx: RunContext[AgentDeps], kolom: str) -> dict:
        """Hoe vaak elke waarde voorkomt. Handig voor ID's (herhaling?) en voor de klassen van de target."""
        return vk.tel_waarden(_zicht(ctx, "tel_waarden", kolom), kolom)

    @ts.tool
    def uitschieters(ctx: RunContext[AgentDeps], kolom: str) -> dict:
        """Waarden buiten de IQR-grenzen (Tukey) in een numerieke kolom."""
        return vk.uitschieters(_zicht(ctx, "uitschieters", kolom), kolom)

    @ts.tool
    def tijdsverloop(ctx: RunContext[AgentDeps], tijd_kolom: str) -> dict:
        """Trend van de target over een tijdkolom. Nodig om te bepalen of je op tijd moet splitsen."""
        return vk.tijdsverloop(_zicht(ctx, "tijdsverloop", tijd_kolom), tijd_kolom, ctx.deps.data.target)

    @ts.tool
    async def vraag_po(ctx: RunContext[AgentDeps], vragen: list[Deelvraag]) -> list[str]:
        """Stel de PO één of meer vragen over betekenis, herkomst of belang. Elke deelvraag krijgt een eigen antwoord.
        Niet voor wat je zelf in de data kunt zien (verdeling, scheefheid, correlatie): gebruik daarvoor je tools."""
        d = ctx.deps
        if fouten := controleer_vragen(vragen, list(d.data.zicht().columns)):
            raise ModelRetry("Pas je vragen aan: " + "; ".join(fouten) + ".")
        ruimte = d.max_vragen - len(d.po_vragen)
        if ruimte <= 0:
            return ["Je hebt het maximum aantal vragen bereikt. Werk verder met wat je weet en meld wat onbekend is."]
        teksten = [q.tekst() for q in vragen[:ruimte]]
        antwoorden = await d.po.vraag_po_meerdere([(t, q.context) for t, q in zip(teksten, vragen)])
        d.po_vragen.extend(zip(teksten, antwoorden))
        uit = [f"{t} → {a}" for t, a in zip(teksten, antwoorden)]
        if len(vragen) > ruimte:
            uit.append(f"{len(vragen) - ruimte} vraag/vragen niet gesteld: maximum bereikt.")
        return uit

    return ts


def train_tools() -> FunctionToolset[AgentDeps]:
    """Verbanden en patronen: alleen na de split, alleen op train."""
    ts: FunctionToolset[AgentDeps] = FunctionToolset()

    @ts.tool
    def na_patroon(ctx: RunContext[AgentDeps], kolom: str) -> dict:
        """Vergelijk rijen mét en zonder NA in `kolom` op alle andere kolommen en de target.
        Verschil in andere kolommen wijst op MAR; alleen in de target: mogelijk MNAR; geen verschil: past bij MCAR."""
        return vk.na_patroon(_zicht(ctx, "na_patroon", kolom, alleen_train=True), kolom, ctx.deps.data.target)

    @ts.tool
    def verdeling(ctx: RunContext[AgentDeps], kolom: str) -> dict:
        """Histogram, gemiddelde, mediaan en scheefheid (getal) of aantallen en zeldzame waarden (categorie)."""
        return vk.verdeling(_zicht(ctx, "verdeling", kolom, alleen_train=True), kolom)

    @ts.tool
    def target_relatie(ctx: RunContext[AgentDeps], kolom: str) -> dict:
        """Samenhang van één kolom met de target (correlatie, of gemiddelden/aandelen per groep of klasse)."""
        return vk.target_relatie(_zicht(ctx, "target_relatie", kolom, alleen_train=True), kolom, ctx.deps.data.target)

    @ts.tool
    def correlatie(ctx: RunContext[AgentDeps]) -> dict:
        """Correlatie van alle numerieke kolommen met de target, en paren die onderling sterk samenhangen (|r| >= 0,8)."""
        return vk.correlatie(_zicht(ctx, "correlatie", alleen_train=True), ctx.deps.data.target)

    return ts
