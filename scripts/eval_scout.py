"""Eval: vindt de Data Scout de valkuilen in de oefendatasets, en klopt de uiteindelijke split?

Gebruik:  uv run python scripts/eval_scout.py [scout-model] [reviewer-model|-] [datasets,komma,gescheiden]
Voorbeeld: uv run python scripts/eval_scout.py google:gemini-flash-latest -
"""
import asyncio
import os
import sys
import time

from dsharness.data import DataLaag
from dsharness.datasets import ALLE, Valkuil
from dsharness.po import ScriptedPO
from dsharness.reviewer import Reviewer
from dsharness.scout import maak_scout
from dsharness.workflow import Team, eerste_blik_tot_split

os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")


def gevonden(v: Valkuil, bevindingen) -> bool:
    for b in bevindingen:
        soort = "target_na" if b.soort == "na" and v.soort == "target_na" else b.soort
        if soort == v.soort and (v.kolom is None or b.kolom == v.kolom):
            return True
    return False


async def een(naam: str, scout_model: str, reviewer_model: str | None) -> dict:
    o = ALLE[naam]()
    po = ScriptedPO(o.po_kennis)
    data = DataLaag(o.df, target=o.target)
    team = Team(scout=maak_scout(scout_model), reviewer=Reviewer(reviewer_model))
    t = time.time()
    u = await eerste_blik_tot_split(po, data, doel=o.doel, taak=o.taak, team=team)
    gemist = [f"{v.soort}({v.kolom or ''})" for v in o.valkuilen if not gevonden(v, u.rapport.bevindingen)]
    return {
        "dataset": naam, "sec": round(time.time() - t),
        "valkuilen": f"{len(o.valkuilen) - len(gemist)}/{len(o.valkuilen)}", "gemist": gemist,
        "po_vragen": len(po.vragen),
        "voorstellen": [(v.spec.type, v.review.oordeel, b.keuze) for v, b in po.voorstellen],
        "split": data.split_actie.methode if data.split_actie else None, "juiste_split": o.juiste_split,
    }


async def main(scout_model: str, reviewer_model: str | None, namen: list[str]) -> None:
    print(f"scout={scout_model} reviewer={reviewer_model or 'alleen regels'}\n")
    for naam in namen:
        try:
            r = await een(naam, scout_model, reviewer_model)
        except Exception as e:  # eval: doorgaan met de volgende dataset
            print(f"{naam}: FOUT {type(e).__name__}: {str(e)[:300]}\n")
            continue
        ok = "OK " if r["split"] == r["juiste_split"] else "MIS"
        print(f"{r['dataset']:8} {r['sec']:>4}s  valkuilen {r['valkuilen']}  split {ok} ({r['split']}, verwacht {r['juiste_split']})"
              f"  PO-vragen {r['po_vragen']}")
        if r["gemist"]:
            print(f"         gemist: {', '.join(r['gemist'])}")
        print(f"         voorstellen: {r['voorstellen']}\n")


if __name__ == "__main__":
    scout = sys.argv[1] if len(sys.argv) > 1 else "google:gemini-flash-latest"
    rev = sys.argv[2] if len(sys.argv) > 2 else scout
    namen = sys.argv[3].split(",") if len(sys.argv) > 3 else list(ALLE)
    asyncio.run(main(scout, None if rev == "-" else rev, namen))
