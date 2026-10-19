"""Eval F3: bepaalt de Verkenner het juiste NA-mechanisme per kolom?

F1 wordt zonder LLM gedaan (de juiste vaste regels en split), zodat alleen F3 gemeten wordt.
Gebruik:  uv run python -u scripts/eval_verkenner.py [model of model+fallback] [datasets,komma,gescheiden]
"""
import asyncio
import sys
import time

from dsharness.data import DataLaag
from dsharness.datasets import ALLE
from dsharness.po import ScriptedPO
from dsharness.reviewer import Reviewer
from dsharness.scout import maak_scout
from dsharness.verkenner import maak_verkenner
from dsharness.workflow import Team, verken_train


async def een(naam: str, model: str) -> None:
    o = ALLE[naam]()
    d = DataLaag(o.df, target=o.target)
    for a in o.juiste_f1:
        d.voer_uit(a)
    po = ScriptedPO(o.po_kennis)
    team = Team(scout=maak_scout(None), reviewer=Reviewer(), verkenner=maak_verkenner(model))
    t = time.time()
    r = await verken_train(po, d, doel=o.doel, taak=o.taak, team=team)
    gekozen = {p.kolom: p.na_mechanisme for p in r.profielen if p.na_aantal}
    goed = sum(gekozen.get(k) == m for k, m in o.na_mechanismen.items())
    print(f"{naam:8} {time.time() - t:>4.0f}s  mechanisme {goed}/{len(o.na_mechanismen)}  PO-vragen {len(po.vragen)}"
          f"  tools {len(d.log)}")
    for k, m in o.na_mechanismen.items():
        p = next(p for p in r.profielen if p.kolom == k)
        teken = "✓" if p.na_mechanisme == m else "✗"
        print(f"         {teken} {k}: {p.na_mechanisme} (echt {m}) · {p.na_onderbouwing[:160]}")
    print(f"         voor F4: {r.voor_f4[:3]}\n")


async def main(model: str, namen: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"verkenner={model}\n")
    for naam in namen:
        try:
            await een(naam, model)
        except Exception as e:  # eval: doorgaan met de volgende dataset
            print(f"{naam}: FOUT {type(e).__name__}: {str(e)[:300]}\n")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "google:gemini-3.1-flash-lite"
    model = model.split("+") if "+" in model else model  # a+b: b is fallback
    namen = sys.argv[2].split(",") if len(sys.argv) > 2 else list(ALLE)
    asyncio.run(main(model, namen))
