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
