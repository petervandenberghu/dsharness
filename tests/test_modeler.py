"""F5: Lab, Reviewer-regels voor F5 en de F5-workflow (geen netwerk)."""
import asyncio

import pytest
from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from dsharness.acties import CVOpzet, Experiment, KiesModel
from dsharness.data import DataLaag
from dsharness.datasets import huizen, klant, patient, winkel
from dsharness.lab import Lab, LabFout
from dsharness.modeler import maak_modeler
from dsharness.po import ScriptedPO
from dsharness.prep_plan import PrepPlan
from dsharness.rapport import ConceptVoorstel
from dsharness.reviewer import ReviewContext, Reviewer
from dsharness.scout import maak_scout
from dsharness.workflow import Team, modelleer

def klaar_voor_f5(oefenset):
    o = oefenset()
    d = DataLaag(o.df, target=o.target)
    for a in o.juiste_f1:
        d.voer_uit(a)
    p = PrepPlan(o.target)
    for s in o.juiste_prep:
        p.voeg_toe(s, d.zicht())
    return o, d, p


# ---------- Lab ----------
def test_lab_volgorde_en_whitelist():
    o, d, p = klaar_voor_f5(huizen)
    lab = Lab(d, p, o.taak)
    with pytest.raises(LabFout, match="CV-opzet"):
        lab.voer_uit(Experiment(naam="b", model="baseline"))
    with pytest.raises(LabFout, match="past niet"):
        lab.voer_uit(CVOpzet(methode="kfold", metriek="accuracy"))
    lab.voer_uit(CVOpzet(methode="kfold", metriek="rmse"))
    with pytest.raises(LabFout, match="baseline"):
        lab.voer_uit(Experiment(naam="rf", model="random_forest"))
    lab.voer_uit(Experiment(naam="b", model="baseline"))
    for e, tekst in [(Experiment(naam="rf", model="random_forest", parameters={"diepte": 3}), "Onbekende"),
                     (Experiment(naam="lin", model="lineair", parameters={"C": 1.0}), "hoort niet"),
                     (Experiment(naam="b", model="lineair"), "al een experiment"),
                     (Experiment(naam="lin", model="lineair", gebalanceerd=True), "classificatie")]:
        with pytest.raises(LabFout, match=tekst):
            lab.voer_uit(e)
    with pytest.raises(LabFout, match="minstens"):
        lab.voer_uit(KiesModel(experiment="b"))
    lab.voer_uit(Experiment(naam="lin", model="lineair", parameters={"alpha": 1.0}))
    assert lab.beste().naam == "lin" and lab.resultaten["lin"].gemiddelde < lab.resultaten["b"].gemiddelde
    with pytest.raises(LabFout, match="ligt vast"):
        lab.voer_uit(CVOpzet(methode="kfold", metriek="mae"))


def test_groep_cv_houdt_patienten_bij_elkaar():
    o, d, p = klaar_voor_f5(patient)
    lab = Lab(d, p, o.taak)
    lab.voer_uit(CVOpzet(methode="groep", folds=4, metriek="roc_auc"))
    df = lab.train_df()
    splitter, groepen = lab._splitter(df)
    for tr, va in splitter.split(df, df[o.target], groepen):
        assert not set(groepen.iloc[tr]) & set(groepen.iloc[va])
        assert df[o.target].iloc[va].nunique() == 2


def test_tijd_cv_traint_op_verleden():
    o, d, p = klaar_voor_f5(winkel)
    lab = Lab(d, p, o.taak)
    lab.voer_uit(CVOpzet(methode="tijd", folds=3, metriek="rmse"))
    df = lab.train_df()
    splitter, _ = lab._splitter(df)
    for tr, va in splitter.split(df):
        assert df["datum"].iloc[tr].max() < df["datum"].iloc[va].min()


def test_prep_zit_in_elke_fold():
    o, d, p = klaar_voor_f5(klant)
    lab = Lab(d, p, o.taak)
    lab.voer_uit(CVOpzet(methode="stratified", metriek="recall"))
    assert list(lab.pipeline(Experiment(naam="x", model="lineair", selectie_k=3), lab.train_df()).named_steps) == ["prep", "selectie", "model"]
    assert lab.positief == "ja"


# ---------- Reviewer F5 ----------
def concept(actie):
    return ConceptVoorstel(actie=actie, observatie="o", onderzocht="o", po_kennis="o", plan="p", waarom="w", bron_sleutels=["cv"])


def review(ctx, actie):
    return asyncio.run(Reviewer().beoordeel(concept(actie), ctx))


def test_reviewer_f5():
    o, d, p = klaar_voor_f5(patient)
    lab = Lab(d, p, o.taak)
    ctx = ReviewContext(data=d, taak=o.taak, plan=p, lab=lab, po_vragen=[("Welke maat is belangrijk?", "roc_auc")])
    assert review(ctx, CVOpzet(methode="stratified", metriek="roc_auc")).oordeel == "fout"  # split was per groep
    assert review(ctx, CVOpzet(methode="groep", folds=4, metriek="roc_auc")).oordeel == "goed"
    ctx.po_vragen.clear()
    assert "businesskeuze" in review(ctx, CVOpzet(methode="groep", folds=4, metriek="roc_auc")).toelichting

    o, d, p = klaar_voor_f5(klant)
    lab = Lab(d, p, o.taak)
    ctx = ReviewContext(data=d, taak=o.taak, plan=p, lab=lab, po_vragen=[("Welke maat?", "recall")])
    assert "Accuracy" in review(ctx, CVOpzet(methode="stratified", metriek="accuracy")).toelichting
    lab.voer_uit(CVOpzet(methode="stratified", metriek="recall"))
    assert review(ctx, Experiment(naam="rf", model="random_forest")).oordeel == "fout"  # nog geen baseline
    lab.voer_uit(Experiment(naam="b", model="baseline"))
    lab.voer_uit(Experiment(naam="lin", model="lineair", gebalanceerd=True))
    r = review(ctx, KiesModel(experiment="lin"))
    assert r.oordeel == "twijfel" and "overfit" in r.toelichting  # train 1,0 tegen validatie ~0,6
    assert "baseline" in review(ctx, KiesModel(experiment="b")).toelichting


# ---------- workflow F5 ----------
def stap(actie: dict) -> dict:
    return {"actie": actie, "observatie": "o", "onderzocht": "o", "po_kennis": "PO: recall telt", "plan": "p",
            "waarom": "w", "bron_sleutels": ["cv"]}


def modeler_script():
    def f(messages, info: AgentInfo) -> ModelResponse:
        delen = [p for m in messages for p in getattr(m, "parts", [])]
        uit = info.output_tools[0].name
        prompt = " ".join(str(p.content) for p in delen if hasattr(p, "content"))
        if "Ronde 2" in prompt:
            return ModelResponse(parts=[ToolCallPart(uit, {"voorstellen": [stap({"type": "kies_model", "experiment": "lin"})]})])
        if not any(isinstance(p, ToolReturnPart) for p in delen):
            return ModelResponse(parts=[ToolCallPart("vraag_po", {"vragen": [{"vraag": "Welke maat is belangrijk: gemiste opzeggers of valse alarmen?"}]})])
        exp = [stap({"type": "experiment", "naam": "lin", "model": "lineair", "parameters": {"C": 0.1}, "gebalanceerd": True}),
               stap({"type": "experiment", "naam": "b", "model": "baseline"})]
        if any(isinstance(p, RetryPromptPart) for p in delen):
            exp.reverse()  # na de terugkoppeling: baseline eerst
        cv = stap({"type": "cv_opzet", "methode": "stratified", "folds": 5, "metriek": "recall"})
        return ModelResponse(parts=[ToolCallPart(uit, {"voorstellen": [cv, *exp]})])
    return FunctionModel(f)


def test_f5_workflow():
    o, d, p = klaar_voor_f5(klant)
    po = ScriptedPO({"maat": "Gemiste opzeggers zijn het ergst; recall telt."})
    team = Team(scout=maak_scout(None), reviewer=Reviewer(), modeler=maak_modeler(modeler_script()))
    lab = asyncio.run(modelleer(po, d, doel=o.doel, taak=o.taak, team=team, plan=p))
    assert lab.gekozen == "lin" and list(lab.resultaten) == ["b", "lin"]
    assert [v.id for v, _ in po.voorstellen] == ["F5-001", "F5-002", "F5-003", "F5-004"]
    assert po.voorstellen[0][0].review.oordeel == "goed"  # PO is gevraagd naar de maat
    assert len(po.resultaten) == 2 and "Gekozen: lin" in po.resultaten[-1]


def test_model_gelijk_aan_baseline_wordt_gesignaleerd():
    o, d, p = klaar_voor_f5(huizen)
    lab = Lab(d, p, o.taak)
    ctx = ReviewContext(data=d, taak=o.taak, plan=p, lab=lab)
    lab.voer_uit(CVOpzet(methode="kfold", metriek="rmse"))
    lab.voer_uit(Experiment(naam="b", model="baseline"))
    voorstel = Experiment(naam="gb", model="gradient_boosting", parameters={"max_depth": 2})
    assert "min_samples_leaf" in review(ctx, voorstel).toelichting  # vooraf: te weinig rijen per fold
    tekst = lab.voer_uit(voorstel)
    assert "gelijk aan de baseline" in tekst and "GELIJK AAN BASELINE" in lab.overzicht()
    assert "Eerder presteerde gradient_boosting" in review(ctx, Experiment(naam="gb2", model="gradient_boosting")).toelichting
    tekst = lab.voer_uit(Experiment(naam="gb3", model="gradient_boosting", parameters={"min_samples_leaf": 3}))
    assert "gelijk aan de baseline" not in tekst and lab.resultaten["gb3"].gemiddelde < lab.resultaten["b"].gemiddelde
