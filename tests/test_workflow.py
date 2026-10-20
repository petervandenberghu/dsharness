"""Hele workflow F1 → F2 → SPLIT met een gescripte Scout (geen netwerk)."""
import asyncio

from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from dsharness.contracts import Besluit
from dsharness.data import DataLaag
from dsharness.datasets import patient
from dsharness.po import ScriptedPO
from dsharness.reviewer import Reviewer
from dsharness.scout import maak_scout
from dsharness.workflow import Team, eerste_blik_tot_split

GLUCOSE = {"actie": {"type": "onmogelijk_naar_na", "kolom": "glucose", "regel": "=0"},
           "observatie": "5 rijen met glucose 0.", "onderzocht": "beschrijf", "po_kennis": "PO: 0 betekent niet gemeten.",
           "plan": "0 wordt NA.", "waarom": "Glucose 0 is onmogelijk.", "bron_sleutels": ["fes_missing"]}
SPLIT_FOUT = {"actie": {"type": "split", "methode": "willekeurig"}, "observatie": "72 rijen.", "onderzocht": "vorm",
              "po_kennis": "Niet gevraagd.", "plan": "Willekeurig splitsen.", "waarom": "Standaard.", "bron_sleutels": ["tts"]}
SPLIT_GOED = {**SPLIT_FOUT, "actie": {"type": "split", "methode": "groep", "groep_kolom": "patient_id"},
              "onderzocht": "tel_waarden(patient_id): elke patiënt 3 keer.", "plan": "Per patiënt splitsen.",
              "waarom": "Dezelfde patiënt mag niet in train én test.", "bron_sleutels": ["cv"]}


def scout_script(herzien: dict):
    """Eerste run: verken en stel een foute split voor. Herziening: lever `herzien`."""
    def f(messages, info: AgentInfo) -> ModelResponse:
        return _stap(messages, info, herzien)
    return FunctionModel(f)


def _stap(messages, info: AgentInfo, herzien: dict) -> ModelResponse:
    uit = info.output_tools[0]
    if "samenvatting" not in uit.parameters_json_schema["properties"]:
        return ModelResponse(parts=[ToolCallPart(uit.name, herzien)])
    if not any(isinstance(p, ToolReturnPart) for m in messages for p in getattr(m, "parts", [])):
        return ModelResponse(parts=[
            ToolCallPart("vorm", {}), ToolCallPart("dubbelen", {}), ToolCallPart("beschrijf", {}),
            ToolCallPart("tel_waarden", {"kolom": "patient_id"}),
            ToolCallPart("vraag_po", {"vragen": [{"kolom": "glucose", "vraag": "Kan glucose 0 zijn?", "context": "5 rijen met 0"}]}),
        ])
    bevindingen = [{"soort": "structuur_groep", "kolom": "patient_id", "beschrijving": "3 bezoeken per patiënt", "bewijs": "tel_waarden"},
                   {"soort": "onmogelijke_waarde", "kolom": "glucose", "beschrijving": "5 keer 0", "bewijs": "beschrijf"}]
    if any(isinstance(p, RetryPromptPart) for m in messages for p in getattr(m, "parts", [])):  # harness: niet volledig
        bevindingen += [{"soort": "na", "kolom": k, "beschrijving": "NA's", "bewijs": "na_telling"} for k in ("bmi", "bloeddruk")]
    return ModelResponse(parts=[ToolCallPart(uit.name, {
        "samenvatting": "Herhaalde patiënten en glucose 0.", "bevindingen": bevindingen, "voorstellen": [GLUCOSE, SPLIT_FOUT]})])


def draai(po, herzien=SPLIT_GOED):
    o = patient()
    data = DataLaag(o.df, target=o.target)
    team = Team(scout=maak_scout(scout_script(herzien)), reviewer=Reviewer())
    uitkomst = asyncio.run(eerste_blik_tot_split(po, data, doel=o.doel, taak=o.taak, team=team))
    return data, uitkomst


def test_reviewer_stuurt_foute_split_terug_en_po_keurt_goede_goed():
    po = ScriptedPO(patient().po_kennis)
    data, uitkomst = draai(po)
    assert po.vragen and "niet gemeten" in po.vragen[0][1]
    ids = [(v.id, v.review.oordeel, b.keuze) for v, b in po.voorstellen]
    assert ids == [("F2-001", "goed", "akkoord"), ("SPLIT-001", "goed", "akkoord")]
    assert data.gesplitst and data.split_actie.methode == "groep"
    assert (data.zicht().glucose == 0).sum() == 0
    assert any("herziet" in m for m in po.meldingen)


def test_po_overrulet_fout():
    """Herziening levert weer een foute split; de PO overrulet met motivatie."""
    def overrulen(v):
        if v.review.oordeel == "fout":
            return Besluit(voorstel_id=v.id, keuze="overrulen", motivatie="Bewust", review_oordeel="fout")
        return Besluit(voorstel_id=v.id, keuze="akkoord", review_oordeel=v.review.oordeel)

    po = ScriptedPO(patient().po_kennis, besluit=overrulen)
    data, _ = draai(po, herzien=SPLIT_FOUT)
    assert [b.keuze for _, b in po.voorstellen] == ["akkoord", "overrulen"]
    assert data.split_actie.methode == "willekeurig"


def test_po_wijst_split_af():
    def afwijzen(v):
        keuze = "afwijzen" if v.spec.type == "split" else "akkoord"
        return Besluit(voorstel_id=v.id, keuze=keuze, review_oordeel=v.review.oordeel)
    po = ScriptedPO(patient().po_kennis, besluit=afwijzen)
    data, uitkomst = draai(po)
    assert not data.gesplitst and not uitkomst.gesplitst
    assert any("Geen split" in m for m in po.meldingen)


def test_volledigheidscontrole():
    from dsharness.rapport import Verkenningsrapport
    from dsharness.scout import ontbrekend
    o = patient()
    data = DataLaag(o.df, target=o.target)
    leeg = Verkenningsrapport(samenvatting="s", bevindingen=[], voorstellen=[SPLIT_GOED])
    mist = " ".join(ontbrekend(leeg, data, o.taak))
    assert "bmi" in mist and "bloeddruk" in mist and "patient_id" in mist


def test_scout_moet_rapport_aanvullen():
    """Het gescripte rapport mist de NA's; de harness stuurt het terug tot het compleet is."""
    po = ScriptedPO(patient().po_kennis)
    data, uitkomst = draai(po)
    soorten = {(b.soort, b.kolom) for b in uitkomst.rapport.bevindingen}
    assert ("na", "bmi") in soorten and ("na", "bloeddruk") in soorten
