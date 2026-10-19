"""Eval F5: kiest de Modeler een CV-opzet die bij de split past, een passende maat, en een verdedigbaar model?

F1 en F4 gaan zonder LLM (juiste stappen uit de oefenset). De gesimuleerde PO zegt akkoord, behalve bij een 'fout'.
Gebruik:  uv run python -u scripts/eval_modeler.py [model of model+fallback] [datasets,komma,gescheiden]
"""
import asyncio
import sys
import time
from collections import Counter

from dsharness.data import DataLaag
from dsharness.datasets import ALLE
from dsharness.modeler import maak_modeler
from dsharness.po import ScriptedPO
from dsharness.prep_plan import PrepPlan
from dsharness.reviewer import Reviewer
from dsharness.scout import maak_scout
from dsharness.workflow import Team, modelleer

VERWACHT_CV = {"willekeurig": "kfold", "gestratificeerd": "stratified", "groep": "groep", "tijd": "tijd"}


async def een(naam: str, model) -> None:
    o = ALLE[naam]()
    d = DataLaag(o.df, target=o.target)
    for a in o.juiste_f1:
        d.voer_uit(a)
    plan = PrepPlan(o.target)
    for s in o.juiste_prep:
        plan.voeg_toe(s, d.zicht())
    po = ScriptedPO(o.po_kennis)
    team = Team(scout=maak_scout(None), reviewer=Reviewer(), modeler=maak_modeler(model))
    t = time.time()
    lab = await modelleer(po, d, doel=o.doel, taak=o.taak, team=team, plan=plan)
    cv_ok = lab.cv is not None and lab.cv.methode == VERWACHT_CV[o.juiste_split]
    oordelen = Counter(v.review.oordeel for v, _ in po.voorstellen)
    afgewezen = sum(b.keuze == "afwijzen" for _, b in po.voorstellen)
    print(f"{naam:8} {time.time() - t:>4.0f}s  CV {'✓' if cv_ok else '✗'} ({lab.cv.methode if lab.cv else '-'}, "
          f"{lab.cv.metriek if lab.cv else '-'})  gekozen: {lab.gekozen or 'GEEN'}  voorstellen {len(po.voorstellen)} "
          f"(reviewer {dict(oordelen)}, afgewezen {afgewezen})  PO-vragen {len(po.vragen)}", flush=True)
    for regel in lab.overzicht().splitlines()[1:]:
        print(f"         {regel}")
    print(flush=True)


async def main(model, namen: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"modeler={model}\n", flush=True)
    for naam in namen:
        try:
            await een(naam, model)
        except Exception as e:  # eval: doorgaan met de volgende dataset
            print(f"{naam}: FOUT {type(e).__name__}: {str(e)[:300]}\n", flush=True)


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "google:gemini-3-flash-preview"
    model = model.split("+") if "+" in model else model
    namen = sys.argv[2].split(",") if len(sys.argv) > 2 else list(ALLE)
    asyncio.run(main(model, namen))
