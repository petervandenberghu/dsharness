"""F4: PrepPlan, Reviewer-regels voor F4 en de F4-workflow (geen netwerk)."""
import asyncio
import pickle

import numpy as np
import pytest
from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from dsharness.acties import Afkappen, Imputeer, LogTransformatie, NAIndicator, OneHot, Schaal, TargetEncoding, VerwijderKolom
from dsharness.contracts import Besluit
from dsharness.data import DataLaag
from dsharness.datasets import huizen, klant, patient
from dsharness.po import ScriptedPO
from dsharness.prep import maak_prep
from dsharness.prep_plan import PrepPlan, PrepPlanFout
from dsharness.rapport import ConceptVoorstel
from dsharness.reviewer import ReviewContext, Reviewer
from dsharness.scout import maak_scout
from dsharness.verkenner import F3Rapport, KolomProfiel
from dsharness.workflow import Team, prepareer


def na_f1(oefenset):
    o = oefenset()
    d = DataLaag(o.df, target=o.target)
    for a in o.juiste_f1:
        d.voer_uit(a)
    return o, d


def f3_van(o, d) -> F3Rapport:
    df = d.zicht()
    prof = [KolomProfiel(kolom=k, type="getal" if df[k].dtype.kind in "fi" else "categorie", na_aantal=int(df[k].isna().sum()),
                         na_mechanisme=o.na_mechanismen.get(k, "MCAR") if df[k].isna().any() else "geen_na", verdeling="v")
            for k in df.columns if k != o.target]
    return F3Rapport(samenvatting="s", target="t", profielen=prof, voor_f4=[])


# ---------- PrepPlan ----------
def test_geen_lek_parameters_alleen_uit_train():
    o, d = na_f1(huizen)
    df = d.zicht()
    p = PrepPlan(o.target)
    for s in [Imputeer(kolom="kamers", methode="mediaan"), Imputeer(kolom="bouwjaar", methode="mediaan"),
              Imputeer(kolom="oppervlakte", methode="mediaan"), OneHot(kolom="wijk"), Schaal()]:
        p.voeg_toe(s, df)
    pipe = p.bouw(df).fit(df[p.features(df)], df[o.target])
    imp = pipe.named_steps["per_kolom"].named_transformers_["kamers"].named_steps["imputeer"]
    assert imp.statistics_[0] == df["kamers"].median()
    # de testset zit in de kluis; de pipeline heeft er nooit naar gekeken
    test = d.eindtest(lambda t: t)
    uit = pipe.transform(test[p.features(df)])
    assert not uit.isna().any().any() and len(uit) == len(test)


def test_volgorde_ligt_vast_in_code():
    o, d = na_f1(huizen)
    df = d.zicht()
    p = PrepPlan(o.target)
    for s in [Schaal(), Imputeer(kolom="oppervlakte", methode="mediaan"), Afkappen(kolom="oppervlakte"),
              Imputeer(kolom="kamers", methode="mediaan"), Imputeer(kolom="bouwjaar", methode="knn"), OneHot(kolom="wijk")]:
        p.voeg_toe(s, df)  # bewust 'verkeerde' volgorde opgegeven
    stappen = p.bouw(df).named_steps["per_kolom"].transformers[0][1].named_steps
    assert list(stappen) == ["afkappen", "imputeer", "schaal"]


def test_ongeldige_stappen_geweigerd():
    o, d = na_f1(huizen)
    df = d.zicht()
    p = PrepPlan(o.target)
    p.voeg_toe(Imputeer(kolom="kamers", methode="mediaan"), df)
    for stap, tekst in [(Imputeer(kolom="wijk", methode="mediaan"), "categorie"), (OneHot(kolom="kamers"), "al numeriek"),
                        (Imputeer(kolom="kamers", methode="gemiddelde"), "al een imputatie"), (OneHot(kolom="prijs_k"), "target"),
                        (VerwijderKolom(kolom="kamers"), "al stappen")]:
        with pytest.raises(PrepPlanFout, match=tekst):
            p.voeg_toe(stap, df)


def test_problemen_en_proef():
    o, d = na_f1(klant)
    df = d.zicht()
    p = PrepPlan(o.target)
    assert len(p.problemen(df)) == 3  # 2 kolommen met NA, 1 categorie
    with pytest.raises(PrepPlanFout):
        p.proef(df)
    for s in [NAIndicator(kolom="inkomen_k"), Imputeer(kolom="inkomen_k", methode="iteratief"),
              Imputeer(kolom="klachten", methode="mediaan"), TargetEncoding(kolom="contract"), Schaal(methode="robuust")]:
        p.voeg_toe(s, df)
    r = p.proef(df)
    assert "inkomen_k_ontbreekt" in r["kolommen"] and r["vangnet"] == ["leeftijd", "maanden_klant"]


def test_nieuwe_data_met_na_en_onbekende_categorie():
    o, d = na_f1(huizen)
    df = d.zicht()
    p = PrepPlan(o.target)
    for s in [Imputeer(kolom="kamers", methode="mediaan"), Imputeer(kolom="bouwjaar", methode="knn"),
              Imputeer(kolom="oppervlakte", methode="mediaan"), OneHot(kolom="wijk")]:
        p.voeg_toe(s, df)
    pipe = p.bouw(df).fit(df[p.features(df)], df[o.target])
    nieuw = df[p.features(df)].head(2).copy()
    nieuw.iloc[0, nieuw.columns.get_loc("afstand_km")] = np.nan
    nieuw.iloc[1, nieuw.columns.get_loc("wijk")] = "Zuid"
    uit = pipe.transform(nieuw)
    assert not uit.isna().any().any()
    assert uit.filter(like="wijk_").iloc[1].sum() == 0
    pickle.dumps(pipe)


# ---------- Reviewer F4 ----------
def concept(actie):
    return ConceptVoorstel(actie=actie, observatie="o", onderzocht="o", po_kennis="o", plan="p", waarom="w", bron_sleutels=["impute"])


def review(o, d, actie, plan=None):
    ctx = ReviewContext(data=d, taak=o.taak, profielen={p.kolom: p for p in f3_van(o, d).profielen}, plan=plan or PrepPlan(o.target))
    return asyncio.run(Reviewer().beoordeel(concept(actie), ctx))


def test_reviewer_f4():
    o, d = na_f1(patient)
    assert review(o, d, Imputeer(kolom="bloeddruk", methode="mediaan")).oordeel == "twijfel"  # MAR, simpel, >10%
    assert "MAR" in review(o, d, Imputeer(kolom="bloeddruk", methode="mediaan")).toelichting
    assert review(o, d, Imputeer(kolom="bloeddruk", methode="knn")).oordeel == "goed"
    assert review(o, d, Imputeer(kolom="geslacht", methode="mediaan")).oordeel == "fout"
    assert review(o, d, LogTransformatie(kolom="leeftijd")).oordeel == "twijfel"
    assert review(o, d, TargetEncoding(kolom="geslacht")).oordeel == "twijfel"
    o, d = na_f1(klant)
    r = review(o, d, Imputeer(kolom="inkomen_k", methode="knn"))
    assert r.oordeel == "twijfel" and "NA-indicator" in r.toelichting  # MNAR zonder indicator
    plan = PrepPlan(o.target)
    plan.voeg_toe(NAIndicator(kolom="inkomen_k"), d.zicht())
    assert review(o, d, Imputeer(kolom="inkomen_k", methode="knn"), plan).oordeel == "goed"


# ---------- workflow F4 ----------
def stap(actie: dict) -> dict:
    return {"actie": actie, "observatie": "o", "onderzocht": "F3", "po_kennis": "niet gevraagd", "plan": "p",
            "waarom": "w", "bron_sleutels": ["impute"]}


COMPLEET = [stap({"type": "imputeer", "kolom": "kamers", "methode": "mediaan"}),
            stap({"type": "imputeer", "kolom": "bouwjaar", "methode": "knn"}),
            stap({"type": "imputeer", "kolom": "oppervlakte", "methode": "mediaan"}),
            stap({"type": "onehot", "kolom": "wijk"}),
            stap({"type": "schaal", "kolommen": "alle", "methode": "standaard"})]


def prep_script():
    """Ronde 1: eerst onvolledig (validator stuurt terug), dan compleet. Ronde 2 (na afwijzing): KNN voor kamers."""
    def f(messages, info: AgentInfo) -> ModelResponse:
        uit = info.output_tools[0].name
        prompt = " ".join(str(p.content) for m in messages for p in getattr(m, "parts", []) if hasattr(p, "content"))
        if "aanvullende stappen" in prompt:
            return ModelResponse(parts=[ToolCallPart(uit, {"voorstellen": [stap({"type": "imputeer", "kolom": "kamers", "methode": "knn"})]})])
        if any(isinstance(p, RetryPromptPart) for m in messages for p in getattr(m, "parts", [])):
            return ModelResponse(parts=[ToolCallPart(uit, {"voorstellen": COMPLEET})])
        return ModelResponse(parts=[ToolCallPart(uit, {"voorstellen": COMPLEET[1:]})])  # kamers vergeten
    return FunctionModel(f)


def test_f4_workflow_afwijzing_en_aanvulling():
    o, d = na_f1(huizen)

    def besluit(v):  # de PO wil geen mediaan voor kamers
        keuze = "afwijzen" if v.actie == "'kamers' imputeren met mediaan" else "akkoord"
        return Besluit(voorstel_id=v.id, keuze=keuze, review_oordeel=v.review.oordeel)

    po = ScriptedPO(o.po_kennis, besluit=besluit)
    team = Team(scout=maak_scout(None), reviewer=Reviewer(), prep=maak_prep(prep_script()))
    plan = asyncio.run(prepareer(po, d, doel=o.doel, taak=o.taak, team=team, f3=f3_van(o, d)))
    assert plan.problemen(d.zicht()) == []
    assert any(isinstance(s, Imputeer) and s.kolom == "kamers" and s.methode == "knn" for s in plan.stappen)
    assert [v.id for v, _ in po.voorstellen][-1] == "F4-006"
    assert len(po.plannen) == 1 and po.plannen[0][1]["rijen"] == d.n_train
    assert any("nog niet model-klaar" in m for m in po.meldingen)
