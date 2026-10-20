"""F7: verslag → notebook (papermill) → HTML, model card en model.joblib (geen netwerk, wel een kernel)."""
import json
from pathlib import Path

import joblib
import nbformat
import pytest

from dsharness.acties import CVOpzet, Experiment, KiesModel
from dsharness.config import Config
from dsharness.contracts import Besluit, ReviewOordeel, Voorstel
from dsharness.datasets import patient
from dsharness.lab import Lab, voer_eindtest
from dsharness.resultaat import maak_resultaat, model_card
from dsharness.verslag import Verslag, maak_verslag, reproduceer
from test_modeler import klaar_voor_f5

ROLLEN = {r: None for r in ("data_scout", "reviewer", "verkenner", "prep_engineer", "modeler")}


def verslag() -> Verslag:
    o, d, p = klaar_voor_f5(patient)
    lab = Lab(d, p, o.taak)
    for a in [CVOpzet(methode="groep", folds=4, metriek="roc_auc"), Experiment(naam="b", model="baseline"),
              Experiment(naam="lin", model="lineair", parameters={"C": 0.1}), KiesModel(experiment="lin")]:
        lab.voer_uit(a)
    voer_eindtest(lab, "lin")
    cfg = Config(project="patient-test", doel=o.doel, target=o.target, taak=o.taak, data="oefenset:patient",
                 data_policy="cloud_allowed", modellen=ROLLEN)
    v_split = Voorstel(id="SPLIT-001", fase="SPLIT", actie="Train/test-split: groep", observatie="o", onderzocht="o",
                       po_kennis="o", plan="p", waarom="w", bronnen=[{"titel": "t", "url": "u"}],
                       review=ReviewOordeel(oordeel="fout", toelichting="t"))
    besluiten = [(v_split, Besluit(voorstel_id="SPLIT-001", keuze="overrulen", motivatie="bewust", review_oordeel="fout"))]
    return maak_verslag(cfg, Path("."), d, besluiten=besluiten, po_vragen=[("Waarom ontbreekt bmi?", "Vergeten.")], plan=p, lab=lab)


def test_reproductie_klopt():
    r = reproduceer(verslag())
    assert r.verschillen == [] and r.lab.eind is not None


def test_model_card_noemt_overrule():
    card = model_card(verslag())
    assert "SPLIT-001" in card and "bewust" in card and "roc_auc" in card


def test_resultaat_notebook(tmp_path):
    paden = maak_resultaat(verslag(), tmp_path)
    nb = nbformat.read(paden["notebook"], 4)
    fouten = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    assert not fouten
    assert "Reproductie klopt" in "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    assert paden["html"].stat().st_size > 10_000 and paden["model"].exists()
    model = joblib.load(paden["model"])
    o = patient()
    voorspelling = model.predict_proba(o.df.drop(columns=[o.target]).head(3))
    assert voorspelling.shape == (3, 2)


def test_vervalst_verslag_laat_notebook_falen(tmp_path):
    v = verslag()
    v.resultaten[1] = v.resultaten[1].model_copy(update={"gemiddelde": 0.99})
    with pytest.raises(Exception, match="experiment 'lin'"):
        maak_resultaat(v, tmp_path)
