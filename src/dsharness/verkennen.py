"""Verkenningsfuncties. Ze geven compacte samenvattingen terug, geen ruwe rijen."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _r(x, n: int = 3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(float(x), n)
    return x if isinstance(x, (int, str, bool)) else str(x)


def _numeriek(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def vorm(df: pd.DataFrame) -> dict:
    kolommen = []
    for k in df.columns:
        s = df[k]
        kolommen.append({
            "kolom": k,
            "type": "getal" if _numeriek(s) else "tekst/categorie",
            "verschillend": int(s.nunique(dropna=True)),
            "voorbeelden": [_r(v) for v in s.dropna().unique()[:3]],
        })
    return {"rijen": len(df), "kolommen": kolommen}


def dubbelen(df: pd.DataFrame) -> dict:
    dup = df.duplicated(keep=False)
    groepen = [list(map(int, g.index)) for _, g in df[dup].groupby(list(df.columns), dropna=False)]
    return {"dubbele_rijen": int(df.duplicated().sum()), "groepen_rijnummers": groepen[:10]}


def na_telling(df: pd.DataFrame) -> dict:
    n = len(df)
    return {"rijen": n, "na_per_kolom": {k: {"aantal": int(v), "fractie": _r(v / n)} for k, v in df.isna().sum().items() if v}}


def beschrijf(df: pd.DataFrame) -> dict:
    num, cat = {}, {}
    for k in df.columns:
        s = df[k]
        if _numeriek(s):
            d = s.describe()
            num[k] = {"n": int(d["count"]), "na": int(s.isna().sum()), "min": _r(d["min"]), "q1": _r(d["25%"]),
                      "mediaan": _r(d["50%"]), "gemiddelde": _r(d["mean"]), "q3": _r(d["75%"]), "max": _r(d["max"]),
                      "scheefheid": _r(s.skew()), "aantal_nul": int((s == 0).sum()), "aantal_negatief": int((s < 0).sum())}
        else:
            vc = s.value_counts()
            cat[k] = {"n": int(s.notna().sum()), "na": int(s.isna().sum()), "verschillend": int(s.nunique()),
                      "meest": _r(vc.index[0]) if len(vc) else None, "meest_aantal": int(vc.iloc[0]) if len(vc) else 0}
    return {"numeriek": num, "categorisch": cat}


def tel_waarden(df: pd.DataFrame, kolom: str, top: int = 15) -> dict:
    vc = df[kolom].value_counts(dropna=False)
    return {
        "kolom": kolom, "rijen": len(df), "verschillend": int(df[kolom].nunique(dropna=False)),
        "top": [{"waarde": "NA" if pd.isna(k) else _r(k), "aantal": int(v), "fractie": _r(v / len(df))} for k, v in vc.head(top).items()],
        "max_herhaling": int(vc.max()) if len(vc) else 0,
    }


def uitschieters(df: pd.DataFrame, kolom: str) -> dict:
    s = df[kolom]
    if not _numeriek(s):
        return {"fout": f"'{kolom}' is geen getal."}
    q1, q3 = s.quantile(.25), s.quantile(.75)
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    buiten = s[(s < lo) | (s > hi)]
    return {"kolom": kolom, "grenzen_iqr": [_r(lo), _r(hi)], "aantal": int(len(buiten)),
            "waarden": [_r(v) for v in buiten.sort_values().tolist()[:10]]}


def tijdsverloop(df: pd.DataFrame, tijd_kolom: str, target: str) -> dict:
    """Trend van de target in de tijd. Mag vóór de split: nodig om de splitswijze te kiezen."""
    d = df[[tijd_kolom, target]].dropna().sort_values(tijd_kolom)
    y = d[target]
    if not _numeriek(y):
        y = (y == y.value_counts().index[-1]).astype(float)
    r = np.corrcoef(np.arange(len(y)), y.to_numpy(dtype=float))[0, 1] if len(y) > 2 else float("nan")
    return {"tijd_kolom": tijd_kolom, "rijen": len(d), "eerste": _r(d[tijd_kolom].iloc[0]), "laatste": _r(d[tijd_kolom].iloc[-1]),
            "correlatie_met_volgorde": _r(r), "uniek_per_rij": bool(d[tijd_kolom].is_unique)}


# ---------- F3: alleen op de trainset ----------
def _id_achtig(s: pd.Series) -> bool:
    """ID's en categorieën met veel waarden geven bij een vergelijking vooral toevalsverschillen."""
    return not _numeriek(s) and (s.nunique() > 10 or "id" in str(s.name).lower())


def _als_getal(y: pd.Series) -> pd.Series:
    """Numerieke target blijft; bij een klasse-target wordt de kleinste klasse 1, de rest 0."""
    if _numeriek(y):
        return y.astype(float)
    return (y == y.value_counts().index[-1]).astype(float).where(y.notna())


def na_patroon(df: pd.DataFrame, kolom: str, target: str) -> dict:
    """Vergelijk rijen mét en zonder NA in `kolom` op alle andere kolommen.

    Getallen: verschil in gemiddelde, in standaardafwijkingen. Categorieën: grootste verschil in aandeel.
    Grote verschillen in andere kolommen wijzen op MAR. Verschilt alleen de target, dan mogelijk MNAR.
    Geen verschil past bij MCAR. MNAR kun je niet bewijzen uit de data.
    """
    rijen = df[df[target].notna()]
    met, zonder = rijen[rijen[kolom].isna()], rijen[rijen[kolom].notna()]
    uit = {"kolom": kolom, "rijen_met_na": len(met), "rijen_zonder_na": len(zonder), "vergelijking": []}
    if len(met) == 0 or len(zonder) == 0:
        uit["conclusie"] = "geen_na" if len(met) == 0 else "alles_na"
        return uit
    sterk_feature = sterk_target = False
    for c in rijen.columns:
        s = rijen[c]
        if c == kolom or _id_achtig(s):
            continue
        is_t = c == target
        if _numeriek(s):
            sd = s.std() or 1.0
            ma, mb = met[c].mean(), zonder[c].mean()
            if pd.isna(ma) or pd.isna(mb):
                continue
            d = abs(ma - mb) / sd
            sterk = d >= (.4 if is_t else .6)
            regel = {"kolom": c, "met_na": _r(ma), "zonder_na": _r(mb), "verschil": f"{d:.2f} sd"}
        else:
            beste = max(s.dropna().unique(), key=lambda k: abs((met[c] == k).mean() - (zonder[c] == k).mean()))
            pa, pb = (met[c] == beste).mean(), (zonder[c] == beste).mean()
            d = abs(pa - pb)
            sterk = d >= (.1 if is_t else .3)
            regel = {"kolom": c, "categorie": _r(beste), "met_na": f"{pa:.0%}", "zonder_na": f"{pb:.0%}", "verschil": f"{d * 100:.0f} %-punt"}
        regel["groot"] = bool(sterk)
        regel["target"] = is_t
        uit["vergelijking"].append(regel)
        sterk_target |= sterk and is_t
        sterk_feature |= sterk and not is_t
    if len(met) < 4:
        uit["conclusie"] = "te_weinig_na_voor_patroon (dan is MCAR de gangbare aanname; vraag de PO)"
    elif sterk_feature:
        uit["conclusie"] = "patroon_in_andere_kolommen (wijst op MAR)"
    elif sterk_target:
        uit["conclusie"] = "alleen_target_verschilt (mogelijk MNAR; alleen de PO kan dat bevestigen)"
    else:
        uit["conclusie"] = "geen_patroon (past bij MCAR; MNAR is niet uit te sluiten zonder de PO)"
    return uit


def verdeling(df: pd.DataFrame, kolom: str, bakken: int = 10) -> dict:
    s = df[kolom].dropna()
    if not _numeriek(s):
        vc = s.value_counts()
        return {"kolom": kolom, "type": "categorie", "aantallen": {str(k): int(v) for k, v in vc.head(15).items()},
                "zeldzaam_minder_dan_3": [str(k) for k, v in vc.items() if v < 3]}
    tellingen, randen = np.histogram(s, bins=bakken)
    return {"kolom": kolom, "type": "getal", "n": len(s), "gemiddelde": _r(s.mean()), "mediaan": _r(s.median()),
            "sd": _r(s.std()), "scheefheid": _r(s.skew()),
            "histogram": [{"van": _r(randen[i]), "tot": _r(randen[i + 1]), "aantal": int(c)} for i, c in enumerate(tellingen)]}


def target_relatie(df: pd.DataFrame, kolom: str, target: str) -> dict:
    d = df[[kolom, target]].dropna()
    x, y = d[kolom], d[target]
    if _numeriek(x) and _numeriek(y):
        return {"kolom": kolom, "pearson": _r(x.corr(y)), "spearman": _r(x.corr(y, method="spearman")), "n": len(d)}
    if _numeriek(y):  # categorische feature, numerieke target
        g = d.groupby(kolom)[target].agg(["mean", "count"])
        return {"kolom": kolom, "gemiddelde_target_per_groep": {str(k): {"gemiddelde": _r(r["mean"]), "n": int(r["count"])} for k, r in g.iterrows()}}
    if _numeriek(x):  # numerieke feature, klasse-target
        g = d.groupby(target)[kolom].agg(["mean", "median", "count"])
        return {"kolom": kolom, "per_klasse": {str(k): {"gemiddelde": _r(r["mean"]), "mediaan": _r(r["median"]), "n": int(r["count"])} for k, r in g.iterrows()}}
    tab = pd.crosstab(x, y, normalize="index")
    return {"kolom": kolom, "aandeel_klasse_per_groep": {str(k): {str(c): f"{v:.0%}" for c, v in r.items()} for k, r in tab.iterrows()},
            "n_per_groep": {str(k): int(v) for k, v in x.value_counts().items()}}


def correlatie(df: pd.DataFrame, target: str) -> dict:
    num = [c for c in df.columns if c != target and _numeriek(df[c])]
    y = _als_getal(df[target])
    met_target = sorted(({"kolom": c, "r": _r(df[c].corr(y))} for c in num), key=lambda r: -abs(r["r"] or 0))
    paren = []
    m = df[num].corr()
    for i, a in enumerate(num):
        for b in num[i + 1:]:
            if abs(m.loc[a, b]) >= .8:
                paren.append({"a": a, "b": b, "r": _r(m.loc[a, b])})
    return {"target": target, "met_target": met_target, "sterk_onderling_r_ge_0_8": paren,
            "let_op": "Correlatie meet alleen rechte-lijnverbanden; een lage r betekent niet dat een feature nutteloos is."}
