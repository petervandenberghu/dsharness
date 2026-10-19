"""Verkenning: welke manier van wachten op de PO werkt in deze kernel?"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from kernelsim import KernelSim  # noqa: E402

SETUP = """
import tempfile, os
from dsharness.cockpit import Cockpit
from dsharness.beslislog import Beslislog
from dsharness.voorbeeld import voorbeeld_voorstel
c = Cockpit(Beslislog(os.path.join(tempfile.mkdtemp(), 'log.jsonl')))
display(c)
"""


def scenario(naam, start_code, na_klik_code=None):
    k = KernelSim()
    try:
        assert k.voer_uit(SETUP), k.errors
        if na_klik_code is None:  # cel blijft lopen tijdens het wachten
            k.start(start_code)
            k.klik("Akkoord")
            klaar = k.pomp(time.monotonic() + 10)
            res = "".join(k.stdout).strip() if klaar else "HANGT (cel kwam niet terug na klik)"
        else:  # cel is klaar, taak loopt op de achtergrond
            assert k.voer_uit(start_code), k.errors
            k.klik("Akkoord")
            time.sleep(0.5)
            k.voer_uit(na_klik_code)
            res = "".join(k.stdout).strip()
        print(f"{naam}: {res} {k.errors or ''}")
    finally:
        k.stop()


import ipykernel  # noqa: E402
print("ipykernel", ipykernel.__version__)
scenario("A achtergrondtaak", "taak = c.start(c.vraag_akkoord(voorbeeld_voorstel()))",
         "print('OK', taak.done() and taak.result().keuze)")
scenario("B await in cel   ", "b = await c.vraag_akkoord(voorbeeld_voorstel()); print('OK', b.keuze)")
