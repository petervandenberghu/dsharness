"""F7: het resultaat. Verslag → notebook (uitgevoerd met papermill) → HTML, plus model.joblib en een model card.

Het notebook reproduceert de hele keten uit het verslag en controleert de getallen. Faalt die controle,
dan faalt het notebook: zo is het resultaat ook een verificatie van de code.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

from .acties import omschrijf
from .verslag import Verslag


def _tabel(kop: list[str], rijen: list[list]) -> str:
    esc = lambda x: str(x).replace("|", "\\|").replace("\n", " ")  # noqa: E731
    return "\n".join(["| " + " | ".join(kop) + " |", "|" + "---|" * len(kop)] + ["| " + " | ".join(esc(c) for c in r) + " |" for r in rijen])


def _besluiten(v: Verslag, fase: str) -> str:
    rijen = [[vs.id, vs.actie, vs.review.oordeel, b.keuze + (f": {b.motivatie}" if b.motivatie else "")]
             for vs, b in v.besluiten if vs.id.startswith(fase)]
    return _tabel(["voorstel", "actie", "Reviewer", "besluit PO"], rijen) if rijen else "_Geen voorstellen in deze fase._"


def model_card(v: Verslag) -> str:
    e, r = v.eind, next((x for x in v.resultaten if x.naam == v.gekozen), None)
    overrule = [f"  - {vs.id} ({vs.actie}): {b.motivatie}" for vs, b in v.besluiten if b.keuze == "overrulen"]
    mnar = [p.kolom for p in (v.f3.profielen if v.f3 else []) if p.na_mechanisme == "MNAR"]
    regels = [
        f"# Model card: {v.project}", "",
        f"**Doel:** {v.doel}", f"**Target:** `{v.target}` ({v.taak})" + (f", positieve klasse '{v.positief}'" if v.positief else ""),
        f"**Gemaakt:** {v.gemaakt:%Y-%m-%d %H:%M} met dsharness · data_policy `{v.data_policy}`", "",
        "## Model", f"- Gekozen experiment: **{v.gekozen}** ({r.omschrijving.split(': ', 1)[-1] if r else '-'})",
        f"- Preparatie: {', '.join(omschrijf(s) for s in v.prep)}",
        f"- Features na preparatie ({len(v.features)}): {', '.join(v.features)}",
        f"- Vangnet (mediaan) voor onverwachte NA in nieuwe data: {', '.join(v.vangnet) or 'geen'}", "",
        "## Prestaties",
        f"- Cross-validatie op train ({v.cv.methode}, {v.cv.folds} folds): {v.cv.metriek} = {r.gemiddelde:.3f} ± {r.sd:.3f}" if r and v.cv else "",
        (f"- Eindtest (eenmalig, {e.n_test} rijen): {e.metriek} = {e.test:.3f}"
         + (f", baseline {e.test_baseline:.3f}" if e.test_baseline is not None else "")
         + ("" if e.binnen_verwachting else " — buiten de verwachting van de cross-validatie")) if e else "- Geen eindtest gedaan.",
        (f"- Overige maten op test: {', '.join(f'{k} {x:.3f}' for k, x in e.extra.items())}" if e else ""), "",
        "## Beperkingen en aandachtspunten",
        f"- Getraind op {e.n_train if e else '?'} rijen; de testset is klein, de schatting dus onzeker.",
        *(["- NA-mechanisme MNAR voor: " + ", ".join(mnar) + " (afgehandeld met een NA-indicator waar gekozen)."] if mnar else []),
        *(["- Besluiten waarin de PO de Reviewer heeft overruled:", *overrule] if overrule else ["- De PO heeft de Reviewer nergens overruled."]),
        "- Alle stappen zijn door de PO goedgekeurd; zie het besluitenlog in het resultaat-notebook.",
    ]
    return "\n".join(x for x in regels if x is not None)


def maak_notebook(v: Verslag) -> nbf.NotebookNode:
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    e = v.eind
    cellen = [
        md(f"# Resultaat: {v.project}\n\n**Doel:** {v.doel}  \n**Target:** `{v.target}` ({v.taak})  \n"
           f"**Gemaakt:** {v.gemaakt:%Y-%m-%d %H:%M} · data_policy `{v.data_policy}`\n\n"
           "Dit notebook is gemaakt door dsharness. Het reproduceert de hele keten uit `verslag.json` en controleert dat "
           "dezelfde getallen terugkomen. Elke stap is door de PO goedgekeurd; de Reviewer beoordeelde elk voorstel vooraf."),
        code("verslag_pad = 'verslag.json'", metadata={"tags": ["parameters"]}),
        code("import pandas as pd\nimport matplotlib.pyplot as plt\nfrom dsharness.verslag import Verslag, reproduceer\n\n"
             "v = Verslag.lees(verslag_pad)\nr = reproduceer(v)\nassert not r.verschillen, r.verschillen\n"
             "print('Reproductie klopt:', r.data)"),
        md("## Samenvatting\n\n" + (f"Gekozen model: **{v.gekozen}**. Eindtest: {e.metriek} = **{e.test:.3f}** op {e.n_test} testrijen"
                                    + (f" (baseline {e.test_baseline:.3f})" if e.test_baseline is not None else "")
                                    + f"; cross-validatie gaf {e.cv_gemiddelde:.3f} ± {e.cv_sd:.3f}." if e else "Nog geen eindtest.")),
        md("## F1 · Eerste blik (vóór de split)\n\n" + (v.f1.samenvatting + "\n\n" + _tabel(
            ["kolom", "soort", "bevinding"], [[b.kolom or "-", b.soort, b.beschrijving] for b in v.f1.bevindingen]) if v.f1 else "")
           + "\n\n### F2 · Vaste regels en split\n\n" + _besluiten(v, "F2") + "\n\n" + _besluiten(v, "SPLIT")),
        md("## F3 · Verkennen op train\n\n" + (v.f3.samenvatting + "\n\n" + _tabel(
            ["kolom", "NA", "mechanisme", "onderbouwing", "verdeling"],
            [[p.kolom, p.na_aantal or "", "" if p.na_mechanisme == "geen_na" else p.na_mechanisme, p.na_onderbouwing, p.verdeling]
             for p in v.f3.profielen]) if v.f3 else "_Geen F3-rapport._")),
        md("## F4 · Preparatie\n\nDe stappen hieronder zitten in één sklearn-Pipeline die alleen op train is gefit.\n\n" + _besluiten(v, "F4")),
        code("df = r.lab.train_df()\nr.plan.bouw(df)"),
        md("## F5 · Modelleren met cross-validatie op train\n\n" + _besluiten(v, "F5")),
        code("res = pd.DataFrame([x.model_dump() for x in r.lab.resultaten.values()])[['naam', 'model', 'gemiddelde', 'sd', 'train']]\n"
             "res = res.sort_values('gemiddelde', ascending=r.lab.cv.metriek in ('rmse', 'mae'))\n"
             "fig, ax = plt.subplots(figsize=(6, 0.4 * len(res) + 1))\n"
             "ax.barh(res['naam'], res['gemiddelde'], xerr=res['sd'], color='#2F6FAF')\n"
             "ax.set_xlabel(f'{r.lab.cv.metriek} (cross-validatie, ± sd)'); ax.invert_yaxis(); plt.tight_layout(); plt.show()\nres"),
        md("## F6 · Eindtest\n\n" + _besluiten(v, "F6")),
        code("e = r.lab.eind\nassert abs(e.test - v.eind.test) < 1e-6\npd.Series({'test': e.test, 'baseline': e.test_baseline, "
             "'cv_gemiddelde': e.cv_gemiddelde, 'cv_sd': e.cv_sd, **e.extra}).round(4)" if e else "print('Geen eindtest')"),
        md("## Model card\n\n" + model_card(v).split("\n", 1)[1]),
        code("import joblib\njoblib.dump(r.lab.eindmodel, 'model.joblib')\nprint('Opgeslagen: model.joblib')" if e else "pass"),
        md("## Vragen aan de PO\n\n" + (_tabel(["vraag", "antwoord"], [[q, a] for q, a in v.po_vragen]) if v.po_vragen else "_Geen._")),
    ]
    nb = nbf.v4.new_notebook(cells=cellen)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def maak_resultaat(v: Verslag, map: str | Path = "resultaat", *, uitvoeren: bool = True) -> dict[str, Path]:
    """Schrijf verslag, model card en notebook; voer het notebook uit (papermill) en exporteer naar HTML."""
    map = Path(map)
    map.mkdir(parents=True, exist_ok=True)
    paden = {"verslag": map / "verslag.json", "model_card": map / "model_card.md", "notebook": map / "07_resultaat.ipynb"}
    v.schrijf(paden["verslag"])
    paden["model_card"].write_text(model_card(v), encoding="utf-8")
    nbf.write(maak_notebook(v), paden["notebook"])
    if uitvoeren:
        import papermill as pm
        from nbconvert import HTMLExporter
        pm.execute_notebook(str(paden["notebook"]), str(paden["notebook"]), cwd=str(map), kernel_name="python3", progress_bar=False)
        html, _ = HTMLExporter().from_filename(str(paden["notebook"]))
        paden["html"] = map / "07_resultaat.html"
        paden["html"].write_text(html, encoding="utf-8")
        paden["model"] = map / "model.joblib"
    return paden
