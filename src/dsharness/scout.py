"""Data Scout: verkent de data in F1 (vóór de split) en stelt vaste regels en een split voor."""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset

from . import verkennen as vk
from .bronnen import catalogus_tekst
from .data import DataLaag
from .modellen import maak_model
from .po import PO
from .rapport import ConceptVoorstel, Verkenningsrapport
from .reviewer import ReviewContext, groep_kandidaten, tijd_kandidaten


@dataclass
class ScoutDeps:
    data: DataLaag
    po: PO
    doel: str
    taak: str
    po_vragen: list[tuple[str, str]] = field(default_factory=list)
    max_vragen: int = 6


INSTRUCTIES = f"""Je bent de Data Scout in een data science-team. Je werkt in fase F1: de eerste blik, vóór de train/test-split.
De gebruiker is de product owner (PO). Alleen de PO weet iets over de business en de herkomst van de data.

Doel van F1: datakwaliteit en structuur vinden, zodat de split goed gekozen kan worden.
1. Verken met je tools: vorm, dubbelen, NA-telling, samenvatting, en waar nodig waarden tellen, uitschieters en tijdsverloop.
2. Onderzoek de structuur: komen ID's vaker voor (meerdere rijen per persoon/klant)? Is er een tijdkolom? Is de target scheef?
3. Vraag de PO (tool vraag_po) als je domeinkennis nodig hebt, bijvoorbeeld: is deze waarde mogelijk? Waarom ontbreekt dit?
   Eén duidelijke vraag per onderwerp, met de getallen erbij. Noem de kolomnaam in je vraag. Ga nooit uit van aannames.
4. Kijk vóór de split NIET naar verbanden tussen features en de target (dat is data snooping). Alleen het tijdsverloop
   van de target en het tellen van de target-klassen mogen, omdat je die nodig hebt om de splitswijze te kiezen.

Lever een rapport met:
- bevindingen: alles wat opvalt, met bewijs (welke tool, welke getallen). NA's in features zijn een bevinding (soort 'na'),
  maar die los je pas na de split op; stel daar nu geen actie voor.
- voorstellen: alleen vaste regels (verwijder_dubbelen, onmogelijk_naar_na, verwijder_zonder_target) en precies één split,
  als laatste. Stel geen actie voor die niets doet. Een waarde is pas 'onmogelijk' als de data of de PO dat laat zien.
  Kies de split op basis van de structuur: groep bij herhaalde ID's, tijd bij een tijdreeks, gestratificeerd bij een scheve
  klasse-target, anders willekeurig. Testset meestal 20-25%.
Vul per voorstel: observatie (met getallen), onderzocht (welke tools), po_kennis (letterlijk wat de PO zei, of dat je het
niet gevraagd hebt), plan, waarom, en bron_sleutels uit deze lijst:
{catalogus_tekst()}
Schrijf alles kort en in eenvoudig Nederlands."""


def _kolom(ctx: RunContext[ScoutDeps], kolom: str) -> None:
    kolommen = list(ctx.deps.data.zicht().columns)
    if kolom not in kolommen:
        raise ModelRetry(f"Kolom '{kolom}' bestaat niet. Kies uit: {kolommen}")


def ontbrekend(rapport: Verkenningsrapport, data: DataLaag, taak: str) -> list[str]:
    """Wat de harness zelf in de data ziet, maar wat niet als bevinding in het rapport staat."""
    gemeld = {(b.soort, b.kolom) for b in rapport.bevindingen}
    df = data.zicht()
    mist = []
    for k in df.columns[df.isna().any()]:
        if ("na", k) not in gemeld and ("target_na", k) not in gemeld:
            mist.append(f"ontbrekende waarden in '{k}' (soort {'target_na' if k == data.target else 'na'})")
    if df.duplicated().any() and not any(s == "dubbelen" for s, _ in gemeld):
        mist.append(f"{int(df.duplicated().sum())} dubbele rij(en) (soort dubbelen)")
    ctx = ReviewContext(data=data, taak=taak)
    for k in groep_kandidaten(ctx):
        if ("structuur_groep", k) not in gemeld:
            mist.append(f"herhaalde waarden in '{k}' (soort structuur_groep)")
    for k in tijd_kandidaten(ctx):
        if ("structuur_tijd", k) not in gemeld:
            mist.append(f"tijdkolom '{k}' (soort structuur_tijd)")
    return mist


@dataclass
class Scout:
    """Twee agents met dezelfde tools: één verkent en rapporteert, één herziet een voorstel."""
    verken: Agent[ScoutDeps, Verkenningsrapport]
    herzie: Agent[ScoutDeps, ConceptVoorstel]


def _opdracht(ctx: RunContext[ScoutDeps]) -> str:
    d = ctx.deps
    return f"Projectdoel: {d.doel}\nTarget: '{d.data.target}' ({d.taak}). Fase: {'na' if d.data.gesplitst else 'vóór'} de split."


def maak_scout(model=None) -> Scout:
    ts: FunctionToolset[ScoutDeps] = FunctionToolset()

    def _zicht(ctx: RunContext[ScoutDeps], tool: str, kolom: str | None = None):
        if kolom is not None:
            _kolom(ctx, kolom)
        ctx.deps.data.noteer(tool, kolom)
        return ctx.deps.data.zicht()

    @ts.tool
    def vorm(ctx: RunContext[ScoutDeps]) -> dict:
        """Aantal rijen, en per kolom het type, het aantal verschillende waarden en 3 voorbeelden."""
        return vk.vorm(_zicht(ctx, "vorm"))

    @ts.tool
    def dubbelen(ctx: RunContext[ScoutDeps]) -> dict:
        """Rijen die in alle kolommen gelijk zijn aan een andere rij."""
        return vk.dubbelen(_zicht(ctx, "dubbelen"))

    @ts.tool
    def na_telling(ctx: RunContext[ScoutDeps]) -> dict:
        """Aantal en fractie ontbrekende waarden per kolom."""
        return vk.na_telling(_zicht(ctx, "na_telling"))

    @ts.tool
    def beschrijf(ctx: RunContext[ScoutDeps]) -> dict:
        """Per numerieke kolom: min, kwartielen, gemiddelde, max, scheefheid, aantal nullen en negatieve waarden.
        Per categorische kolom: aantal verschillende waarden en de meest voorkomende."""
        return vk.beschrijf(_zicht(ctx, "beschrijf"))

    @ts.tool
    def tel_waarden(ctx: RunContext[ScoutDeps], kolom: str) -> dict:
        """Hoe vaak elke waarde voorkomt. Handig voor ID's (herhaling?) en voor de klassen van de target."""
        return vk.tel_waarden(_zicht(ctx, "tel_waarden", kolom), kolom)

    @ts.tool
    def uitschieters(ctx: RunContext[ScoutDeps], kolom: str) -> dict:
        """Waarden buiten de IQR-grenzen (Tukey) in een numerieke kolom."""
        return vk.uitschieters(_zicht(ctx, "uitschieters", kolom), kolom)

    @ts.tool
    def tijdsverloop(ctx: RunContext[ScoutDeps], tijd_kolom: str) -> dict:
        """Trend van de target over een tijdkolom. Nodig om te bepalen of je op tijd moet splitsen."""
        return vk.tijdsverloop(_zicht(ctx, "tijdsverloop", tijd_kolom), tijd_kolom, ctx.deps.data.target)

    @ts.tool
    async def vraag_po(ctx: RunContext[ScoutDeps], vraag: str, context: str = "") -> str:
        """Stel de PO een vraag over de business of de data. `context`: de getallen die je zag."""
        d = ctx.deps
        if len(d.po_vragen) >= d.max_vragen:
            return "Je hebt het maximum aantal vragen bereikt. Werk verder met wat je weet en meld wat onbekend is."
        antwoord = await d.po.vraag_po(vraag, context)
        d.po_vragen.append((vraag, antwoord))
        return antwoord

    m = maak_model(model)
    verken = Agent(m, deps_type=ScoutDeps, output_type=Verkenningsrapport, instructions=[INSTRUCTIES, _opdracht],
                   toolsets=[ts], name="data_scout", retries=3)
    herzie = Agent(m, deps_type=ScoutDeps, output_type=ConceptVoorstel, instructions=[INSTRUCTIES, _opdracht],
                   toolsets=[ts], name="data_scout_herzien", retries=3)

    @verken.output_validator
    def _volledig(ctx: RunContext[ScoutDeps], uit: Verkenningsrapport) -> Verkenningsrapport:
        if mist := ontbrekend(uit, ctx.deps.data, ctx.deps.taak):
            raise ModelRetry("Je rapport is niet volledig. Voeg een bevinding toe (met bewijs) voor: " + "; ".join(mist)
                             + ". Verken waar nodig eerst met je tools.")
        return uit

    return Scout(verken=verken, herzie=herzie)
