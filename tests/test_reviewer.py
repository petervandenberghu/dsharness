import asyncio

from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.messages import ModelResponse, ToolCallPart

from dsharness.acties import OnmogelijkNaarNA, Split, VerwijderDubbelen
from dsharness.data import DataLaag
from dsharness.datasets import huizen, klant, patient, winkel
from dsharness.rapport import ConceptVoorstel
from dsharness.reviewer import ReviewContext, Reviewer


def concept(actie):
    return ConceptVoorstel(actie=actie, observatie="o", onderzocht="o", po_kennis="o", plan="p", waarom="w", bron_sleutels=["crisp"])


def oordeel(actie, ctx, model=None):
    return asyncio.run(Reviewer(model).beoordeel(concept(actie), ctx))


def ctx_van(oefenset):
    o = oefenset()
    return ReviewContext(data=DataLaag(o.df, target=o.target), taak=o.taak)


def test_willekeurige_split_bij_patienten_is_fout():
    r = oordeel(Split(methode="willekeurig"), ctx_van(patient))
    assert r.oordeel == "fout" and "patient_id" in r.toelichting


def test_willekeurige_split_bij_tijdreeks_is_fout():
    assert oordeel(Split(methode="willekeurig"), ctx_van(winkel)).oordeel == "fout"


def test_groep_split_onderbouwd_is_goed():
    ctx = ctx_van(patient)
    ctx.data.noteer("tel_waarden", "patient_id")
    assert oordeel(Split(methode="groep", groep_kolom="patient_id"), ctx).oordeel == "goed"


def test_groep_split_niet_onderbouwd_is_twijfel():
    r = oordeel(Split(methode="groep", groep_kolom="patient_id"), ctx_van(patient))
    assert r.oordeel == "twijfel" and "Niet onderbouwd" in r.toelichting


def test_scheve_target_willekeurig_is_twijfel():
    assert oordeel(Split(methode="willekeurig"), ctx_van(klant)).oordeel == "twijfel"


def test_split_met_dubbelen_is_twijfel():
    r = oordeel(Split(methode="willekeurig"), ctx_van(huizen))
    assert r.oordeel == "twijfel" and "dubbele" in r.toelichting


def test_onmogelijk_naar_na_vraagt_onderzoek_en_po():
    ctx = ctx_van(huizen)
    r = oordeel(OnmogelijkNaarNA(kolom="oppervlakte", regel="<0"), ctx)
    assert r.oordeel == "twijfel" and "Niet onderzocht" in r.toelichting and "PO" in r.toelichting
    ctx.data.noteer("beschrijf")
    ctx.po_vragen.append(("Kan oppervlakte negatief zijn?", "Nee, tikfout."))
    assert oordeel(OnmogelijkNaarNA(kolom="oppervlakte", regel="<0"), ctx).oordeel == "goed"


def test_dubbelen_na_split_is_minstens_twijfel():
    ctx = ctx_van(huizen)
    ctx.data.noteer("dubbelen")
    ctx.data.voer_uit(Split(methode="willekeurig"))
    assert oordeel(VerwijderDubbelen(), ctx).oordeel in ("twijfel", "fout")


def _llm(oordeel_: str):
    def f(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"oordeel": oordeel_, "toelichting": "LLM zegt " + oordeel_})])
    return FunctionModel(f)


def test_llm_kan_niet_milder_maken():
    r = oordeel(Split(methode="willekeurig"), ctx_van(patient), model=_llm("goed"))
    assert r.oordeel == "fout"


def test_llm_kan_strenger_maken():
    ctx = ctx_van(patient)
    ctx.data.noteer("tel_waarden", "patient_id")
    r = oordeel(Split(methode="groep", groep_kolom="patient_id"), ctx, model=_llm("twijfel"))
    assert r.oordeel == "twijfel" and "LLM zegt twijfel" in r.toelichting


def test_llm_valt_weg_dan_alleen_regels():
    from pydantic_ai.exceptions import ModelHTTPError

    def plat(messages, info):
        raise ModelHTTPError(400, "gemini-x", body="kapot")  # niet tijdelijk: geen retry, direct fout
    ctx = ctx_van(patient)
    r = oordeel(Split(methode="willekeurig"), ctx, model=FunctionModel(plat))
    assert r.oordeel == "fout" and "Alleen de vaste regels" in r.toelichting and "gemini-x" in r.toelichting


def test_korte_fout_en_foutkaart():
    from pydantic_ai.exceptions import FallbackExceptionGroup, ModelHTTPError

    from dsharness.cockpit import foutkaart
    from dsharness.modellen import LimietBereikt, korte_fout
    groep = FallbackExceptionGroup("alle modellen", [LimietBereikt("gemini-a", "wacht 16 uur"),
                                                    ModelHTTPError(503, "gemini-b", body="druk")])
    assert korte_fout(groep) == "gemini-a: daglimiet bereikt; gemini-b: overbelast (503)"
    h = foutkaart(groep)
    assert "niet beschikbaar" in h and "<details>" in h and "project.yaml" in h
