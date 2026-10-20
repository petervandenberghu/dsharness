"""End-to-end: de hele keten F1 → F7 met echte modellen uit project.yaml en een gesimuleerde PO.

Gebruik:  uv run python -u scripts/e2e.py [project.yaml]
De gesimuleerde PO kent de oefenset uit project.yaml (data: oefenset:<naam>) en zegt akkoord, behalve bij 'fout'.
"""
import asyncio
import sys
import time
from collections import Counter

from dsharness.datasets import ALLE
from dsharness.harness import Harness
from dsharness.po import ScriptedPO


class LoggendePO(ScriptedPO):
    def melding(self, tekst: str) -> None:
        super().melding(tekst)
        print(f"  · {tekst[:160]}", flush=True)


async def main(pad: str) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    h = Harness(pad)
    o = ALLE[h.config.data.split(":", 1)[1]]()
    po = LoggendePO(o.po_kennis)
    t = time.time()
    try:
        await h.run(po)
    finally:
        print(f"\nDuur: {time.time() - t:.0f}s · PO-vragen: {len(po.vragen)} · voorstellen: {len(po.voorstellen)} "
              f"· Reviewer: {dict(Counter(v.review.oordeel for v, _ in po.voorstellen))} "
              f"· besluiten: {dict(Counter(b.keuze for _, b in po.voorstellen))}")
        print(f"Fase bereikt: split={h.data.gesplitst}, F3={h.f3 is not None}, F4={h.plan is not None}, "
              f"F5 gekozen={getattr(h.lab, 'gekozen', None)}, F6={getattr(h.lab, 'eind', None) is not None}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "project.yaml"))
