"""F6: eenmalige eindtest (geen netwerk)."""
import asyncio

import pytest

from dsharness.acties import CVOpzet, Eindtest, Experiment, KiesModel
from dsharness.contracts import Besluit
from dsharness.data import KluisFout
from dsharness.datasets import huizen, klant
from dsharness.lab import Lab, LabFout, voer_eindtest
from dsharness.po import ScriptedPO
from dsharness.rapport import ConceptVoorstel
from dsharness.reviewer import ReviewContext, Reviewer
from dsharness.scout import maak_scout
from dsharness.workflow import Team, eindtest
from test_modeler import klaar_voor_f5


def gekozen_lab(oefenset, cv):
    o, d, p = klaar_voor_f5(oefenset)
    lab = Lab(d, p, o.taak)
    for a in [cv, Experiment(naam="b", model="baseline"), Experiment(naam="lin", model="lineair"), KiesModel(experiment="lin")]:
        lab.voer_uit(a)
    return o, d, lab


def test_eindtest_na_akkoord_een_keer():
    o, d, lab = gekozen_lab(huizen, CVOpzet(methode="kfold", metriek="rmse"))
    po = ScriptedPO()
    asyncio.run(eindtest(po, d, taak=o.taak, team=Team(scout=maak_scout(None), reviewer=Reviewer()), lab=lab))
    [(v, b)] = po.voorstellen
    assert v.id == "F6-001" and v.review.oordeel == "goed" and b.keuze == "akkoord"
    assert d.test_gebruikt and lab.eind.n_test == d.n_test and lab.eindmodel is not None
    assert lab.eind.test < lab.eind.test_baseline  # rmse: lager is beter
    assert set(lab.eind.extra) == {"rmse", "mae", "r2"} and po.eindtesten == [lab.eind]
    with pytest.raises(KluisFout):
        voer_eindtest(lab, "lin")


def test_afwijzen_laat_testset_ongemoeid():
    o, d, lab = gekozen_lab(klant, CVOpzet(methode="stratified", metriek="recall"))
    po = ScriptedPO(besluit=lambda v: Besluit(voorstel_id=v.id, keuze="afwijzen", review_oordeel=v.review.oordeel))
    asyncio.run(eindtest(po, d, taak=o.taak, team=Team(scout=maak_scout(None), reviewer=Reviewer()), lab=lab))
    assert not d.test_gebruikt and lab.eind is None


def test_reviewer_f6():
    o, d, lab = gekozen_lab(huizen, CVOpzet(methode="kfold", metriek="rmse"))
    ctx = ReviewContext(data=d, taak=o.taak, plan=lab.plan, lab=lab)
    c = lambda e: ConceptVoorstel(actie=Eindtest(experiment=e), observatie="o", onderzocht="o", po_kennis="o",  # noqa: E731
                                  plan="p", waarom="w", bron_sleutels=["geron"])
    assert asyncio.run(Reviewer().beoordeel(c("b"), ctx)).oordeel == "fout"
    assert asyncio.run(Reviewer().beoordeel(c("lin"), ctx)).oordeel == "goed"
    with pytest.raises(LabFout):
        voer_eindtest(lab, "b")
    voer_eindtest(lab, "lin")
    assert "al gebruikt" in asyncio.run(Reviewer().beoordeel(c("lin"), ctx)).toelichting
