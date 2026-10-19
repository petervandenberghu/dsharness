"""Cockpit in een echte kernel, met nagebootste klikken uit de frontend."""
import time

import pytest

from kernelsim import KernelSim

SETUP = """
import tempfile, os
from dsharness.cockpit import Cockpit
from dsharness.beslislog import Beslislog
from dsharness.voorbeeld import voorbeeld_voorstel
log = Beslislog(os.path.join(tempfile.mkdtemp(), 'log.jsonl'))
c = Cockpit(log)
display(c)
"""


@pytest.fixture
def k():
    sim = KernelSim()
    assert sim.voer_uit(SETUP), sim.errors
    yield sim
    sim.stop()


def uitkomst(k, code):
    time.sleep(0.5)
    k.stdout.clear()
    assert k.voer_uit(code), k.errors
    return "".join(k.stdout).strip()


def test_akkoord(k):
    assert k.voer_uit("taak = c.start(c.vraag_akkoord(voorbeeld_voorstel('twijfel')))")
    k.klik("Akkoord")
    assert uitkomst(k, "print(taak.result().keuze, len(log.lees()))") == "akkoord 1"


def test_fout_akkoord_geweigerd_overrulen_met_motivatie(k):
    assert k.voer_uit("taak = c.start(c.vraag_akkoord(voorbeeld_voorstel('fout')))")
    k.klik("Akkoord")  # knop is in de UI uitgeschakeld; ook een geforceerde klik mag niets opleveren
    assert uitkomst(k, "print(taak.done())") == "False"
    k.klik("Overrulen")
    k.klik("Bevestigen")  # zonder motivatie: geweigerd
    assert uitkomst(k, "print(taak.done())") == "False"
    k.typ("Op deze dataset is er geen aparte testperiode")
    k.klik("Bevestigen")
    assert uitkomst(k, "b = taak.result(); print(b.keuze, '|', b.motivatie)") == \
        "overrulen | Op deze dataset is er geen aparte testperiode"


def test_vraag_po(k):
    assert k.voer_uit("taak = c.start(c.vraag_po('Waarom ontbreekt aanrijtijd soms?'))")
    k.wacht_op_knop("Versturen")
    k.typ("Weet ik niet")
    k.klik("Versturen")
    assert uitkomst(k, "print(taak.result())") == "Weet ik niet"


def test_tweede_klik_na_besluit_wordt_genegeerd(k):
    assert k.voer_uit("taak = c.start(c.vraag_akkoord(voorbeeld_voorstel('goed')))")
    k.klik("Akkoord")
    k.klik("Akkoord")  # geforceerd; de knop is in de UI al uitgeschakeld
    k.klik("Afwijzen")
    k.klik("Bevestigen")
    assert uitkomst(k, "print(taak.result().keuze, len(log.lees()))") == "akkoord 1"


def test_rapportkaarten(k):
    code = """
from dsharness.verkenner import F3Rapport, KolomProfiel
from dsharness.rapport import Verkenningsrapport, Bevinding
r3 = F3Rapport(samenvatting="s <b>", target="t", voor_f4=["bouwjaar: MAR"], profielen=[
    KolomProfiel(kolom="bouwjaar", type="getal", na_aantal=4, na_mechanisme="MAR", na_onderbouwing="patroon in wijk", verdeling="v"),
    KolomProfiel(kolom="wijk", type="categorie", na_aantal=0, na_mechanisme="geen_na", verdeling="4 groepen")])
c.toon_rapport(r3)
print(len(c.feed.children), 'bouwjaar' in c.feed.children[-1].value, '&lt;b&gt;' in c.feed.children[-1].value, [f for f, _ in log.lees_rapporten()])
"""
    k.stdout.clear()
    assert k.voer_uit(code), k.errors
    assert "".join(k.stdout).strip() == "1 True True ['F3']"


def test_plankaart(k):
    code = """
from dsharness.datasets import huizen
from dsharness.data import DataLaag
from dsharness.prep_plan import PrepPlan
from dsharness.acties import Imputeer, OneHot, Schaal
o = huizen(); d = DataLaag(o.df, target=o.target)
for a in o.juiste_f1: d.voer_uit(a)
df = d.zicht(); p = PrepPlan(o.target)
for s in [Imputeer(kolom="kamers", methode="mediaan"), Imputeer(kolom="bouwjaar", methode="knn"),
          Imputeer(kolom="oppervlakte", methode="mediaan"), OneHot(kolom="wijk"), Schaal()]:
    p.voeg_toe(s, df)
c.toon_plan(p, p.proef(df))
html = c.feed.children[-1].value
print('Model-klaar' in html, 'afstand_km' in html, [f for f, _ in log.lees_rapporten()])
"""
    k.stdout.clear()
    assert k.voer_uit(code), k.errors
    assert "".join(k.stdout).strip() == "True True ['F4']"
