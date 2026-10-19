"""Eval F4: maakt de Prep Engineer een model-klare pipeline die past bij de NA-mechanismen?

F1 gaat zonder LLM (juiste vaste regels en split); de F3-profielen komen uit de ground truth.
De gesimuleerde PO zegt akkoord, behalve bij een 'fout' van de Reviewer.
Gebruik:  uv run python -u scripts/eval_prep.py [model of model+fallback] [datasets,komma,gescheiden]
"""
import asyncio
import sys
import time
from collections import Counter

from dsharness.acties import Imputeer, NAIndicator, omschrijf
from dsharness.data import DataLaag
from dsharness.datasets import ALLE
from dsharness.po import ScriptedPO
from dsharness.prep import maak_prep
from dsharness.reviewer import Reviewer
from dsharness.scout import maak_scout
from dsharness.verkenner import F3Rapport, KolomProfiel
from dsharness.workflow import Team, prepareer


def f3_uit_ground_truth(o, d) -> F3Rapport:
    df = d.zicht()
    prof = [KolomProfiel(kolom=k, type="getal" if df[k].dtype.kind in "fi" else "categorie", na_aantal=int(df[k].isna().sum()),
                         na_mechanisme=o.na_mechanismen.get(k, "MCAR") if df[k].isna().any() else "geen_na",
                         na_onderbouwing=o.po_kennis.get(k, ""), verdeling=f"scheefheid {df[k].skew():.1f}" if df[k].dtype.kind in "fi" else
                         f"{df[k].nunique()} categorieën")
            for k in df.columns if k != o.target]
    return F3Rapport(samenvatting="ground truth", target=o.target, profielen=prof, voor_f4=[])


async def een(naam: str, model) -> None:
    o = ALLE[naam]()
    d = DataLaag(o.df, target=o.target)
    for a in o.juiste_f1:
        d.voer_uit(a)
    po = ScriptedPO(o.po_kennis)
    team = Team(scout=maak_scout(None), reviewer=Reviewer(), prep=maak_prep(model))
    t = time.time()
    plan = await prepareer(po, d, doel=o.doel, taak=o.taak, team=team, f3=f3_uit_ground_truth(o, d))
    klaar = not plan.problemen(d.zicht()) and bool(po.plannen)
    checks = []
    for k, m in o.na_mechanismen.items():
        imp = next((s for s in plan.stappen if isinstance(s, Imputeer) and s.kolom == k), None)
        ind = any(isinstance(s, NAIndicator) and s.kolom == k for s in plan.stappen)
        ok = imp is not None and (m != "MAR" or imp.methode in ("knn", "iteratief")) and (m != "MNAR" or ind)
        checks.append(f"{'✓' if ok else '✗'} {k} [{m}]: {imp.methode if imp else 'geen imputatie'}{' + indicator' if ind else ''}")
    oordelen = Counter(v.review.oordeel for v, _ in po.voorstellen)
    print(f"{naam:8} {time.time() - t:>4.0f}s  model-klaar {'ja' if klaar else 'NEE'}  voorstellen {len(po.voorstellen)} "
          f"(reviewer {dict(oordelen)})  PO-vragen {len(po.vragen)}", flush=True)
    for c in checks:
        print(f"         {c}")
    print("         plan: " + " | ".join(omschrijf(s) for s in plan.stappen) + "\n", flush=True)


async def main(model, namen: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"prep_engineer={model}\n", flush=True)
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
