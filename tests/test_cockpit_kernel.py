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
