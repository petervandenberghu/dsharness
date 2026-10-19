"""Tools voor de agents, als toolsets die meerdere agents kunnen delen.

Elke tool leest via DataLaag.zicht() en wordt genoteerd in data.log, zodat de harness kan controleren wat echt onderzocht is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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

    def po_gevraagd_over(self, kolom: str) -> bool:
        return any(kolom.lower() in (v + " " + a).lower() for v, a in self.po_vragen)


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
    async def vraag_po(ctx: RunContext[AgentDeps], vraag: str, context: str = "") -> str:
        """Stel de PO een vraag over de business of de data. Noem de kolomnaam. `context`: de getallen die je zag."""
        d = ctx.deps
        if len(d.po_vragen) >= d.max_vragen:
            return "Je hebt het maximum aantal vragen bereikt. Werk verder met wat je weet en meld wat onbekend is."
        antwoord = await d.po.vraag_po(vraag, context)
        d.po_vragen.append((vraag, antwoord))
        return antwoord

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
