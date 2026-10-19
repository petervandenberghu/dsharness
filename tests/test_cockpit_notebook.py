"""Speelt de demo in examples/cockpit_demo.ipynb helemaal na: PO-vraag, akkoord, overrulen."""
import os
from pathlib import Path

import nbformat

from kernelsim import KernelSim

NB = Path(__file__).resolve().parents[1] / "examples" / "cockpit_demo.ipynb"


def test_demo_route(tmp_path):
    code = [c.source for c in nbformat.read(NB, 4).cells if c.cell_type == "code"]
    k = KernelSim()
    try:
        assert k.voer_uit(f"import os; os.chdir({str(tmp_path)!r})")
        for src in code[:3]:
            assert k.voer_uit(src), k.errors
        k.wacht_op_knop("Versturen"); k.typ("Weet ik niet"); k.klik("Versturen")
        k.klik("Akkoord")
        k.klik("Overrulen"); k.typ("Ter demonstratie"); k.klik("Bevestigen")
        import time; time.sleep(0.5)
        k.stdout.clear()
        assert k.voer_uit("print(taak.done(), [(v.id, b.keuze) for v, b in log.lees()])"), k.errors
        assert "".join(k.stdout).strip() == "True [('F4-007', 'akkoord'), ('F2-003', 'overrulen')]"
    finally:
        k.stop()
