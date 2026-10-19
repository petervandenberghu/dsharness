"""PrepPlan: goedgekeurde F4-stappen → één sklearn-Pipeline, alleen te fitten op train.

De volgorde ligt vast in code, per kolom: afkappen → log → imputeren → schalen; daarna (optioneel) knn/iteratief
imputeren over alle numerieke kolommen; tot slot encoders. Zo kan 'schalen vóór imputeren' niet voorkomen.
Numerieke kolommen zonder imputatie krijgen een zichtbaar vangnet (mediaan), voor onverwachte NA in nieuwe data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler, TargetEncoder
from sklearn.utils.validation import check_is_fitted, validate_data

from .acties import Afkappen, Imputeer, LogTransformatie, NAIndicator, OneHot, Schaal, TargetEncoding, VerwijderKolom, omschrijf

EENVOUDIG = {"gemiddelde": "mean", "mediaan": "median", "modus": "most_frequent", "constante": "constant"}
SCHALERS = {"standaard": StandardScaler, "minmax": MinMaxScaler, "robuust": RobustScaler}


class PrepPlanFout(ValueError):
    """Een stap past niet in het plan (bestaat niet, verkeerd type, dubbel)."""


class IQRAfkapper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Kapt af op Q1 - 1,5·IQR en Q3 + 1,5·IQR, geleerd bij fit. NA blijft NA."""

    def fit(self, X, y=None):
        X = validate_data(self, X, ensure_all_finite="allow-nan", dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        self.lo_, self.hi_ = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, ensure_all_finite="allow-nan", dtype=float, reset=False)
        return np.clip(X, self.lo_, self.hi_)


def is_na(X):
    """NA-indicator als 0/1 (module-niveau, zodat de pipeline te pickelen is)."""
    return pd.DataFrame(X).isna().astype(int)


def na_namen(_transformer, namen):
    return [f"{n}_ontbreekt" for n in namen]


def _numeriek(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


class PrepPlan:
    def __init__(self, target: str):
        self.target = target
        self.stappen: list = []

    # ---------- opbouwen ----------
    def verwijderd(self) -> set[str]:
        return {s.kolom for s in self.stappen if isinstance(s, VerwijderKolom)}

    def features(self, df: pd.DataFrame) -> list[str]:
        return [k for k in df.columns if k != self.target and k not in self.verwijderd()]

    def _van(self, kolom: str, *types) -> list:
        return [s for s in self.stappen if isinstance(s, types) and getattr(s, "kolom", None) == kolom]

    def geschaald(self, kolom: str, df: pd.DataFrame):
        for s in self.stappen:
            if isinstance(s, Schaal) and (kolom in s.kolommen if isinstance(s.kolommen, list) else _numeriek(df[kolom])):
                return s
        return None

    def multi_imputatie(self) -> str | None:
        m = {s.methode for s in self.stappen if isinstance(s, Imputeer) and s.methode in ("knn", "iteratief")}
        return m.pop() if m else None

    def voeg_toe(self, stap, df: pd.DataFrame) -> str:
        self._valideer(stap, df)
        self.stappen.append(stap)
        return f"Toegevoegd aan de pipeline: {omschrijf(stap)}."

    def _valideer(self, stap, df: pd.DataFrame) -> None:
        feats = self.features(df)
        kolommen = stap.kolommen if isinstance(stap, Schaal) and isinstance(stap.kolommen, list) else (
            [stap.kolom] if hasattr(stap, "kolom") else [])
        for k in kolommen:
            if k == self.target:
                raise PrepPlanFout(f"'{k}' is de target; die hoort niet in de preparatie.")
            if k not in feats:
                raise PrepPlanFout(f"Kolom '{k}' bestaat niet of is al verwijderd.")
        num = {k: _numeriek(df[k]) for k in kolommen}
        match stap:
            case VerwijderKolom(kolom=k):
                if any(getattr(s, "kolom", None) == k for s in self.stappen):
                    raise PrepPlanFout(f"Er staan al stappen voor '{k}'; verwijder die eerst.")
            case NAIndicator(kolom=k):
                if self._van(k, NAIndicator):
                    raise PrepPlanFout(f"'{k}' heeft al een NA-indicator.")
            case Imputeer(kolom=k, methode=m):
                if self._van(k, Imputeer):
                    raise PrepPlanFout(f"'{k}' heeft al een imputatie.")
                if not num[k] and m in ("gemiddelde", "mediaan", "knn", "iteratief"):
                    raise PrepPlanFout(f"'{k}' is een categorie; gebruik modus of een constante (bijv. 'onbekend').")
                if (andere := self.multi_imputatie()) and m in ("knn", "iteratief") and m != andere:
                    raise PrepPlanFout(f"Er is al {andere}-imputatie; gebruik één multi-kolommethode.")
            case Afkappen(kolom=k) | LogTransformatie(kolom=k):
                if not num[k]:
                    raise PrepPlanFout(f"'{k}' is geen getal.")
                if self._van(k, type(stap)):
                    raise PrepPlanFout(f"Deze stap staat al in het plan voor '{k}'.")
                if isinstance(stap, LogTransformatie) and (df[k] < 0).any():
                    raise PrepPlanFout(f"'{k}' bevat negatieve waarden; log(x + 1) kan dan niet.")
            case OneHot(kolom=k) | TargetEncoding(kolom=k):
                if num[k]:
                    raise PrepPlanFout(f"'{k}' is al numeriek; encoderen is niet nodig.")
                if self._van(k, OneHot, TargetEncoding):
                    raise PrepPlanFout(f"'{k}' heeft al een encoding.")
            case Schaal(kolommen=ks):
                doel = ks if isinstance(ks, list) else [k for k in feats if _numeriek(df[k])]
                if bad := [k for k in doel if not _numeriek(df[k])]:
                    raise PrepPlanFout(f"Alleen numerieke kolommen kun je schalen, niet: {bad}.")
                if dubbel := [k for k in doel if self.geschaald(k, df)]:
                    raise PrepPlanFout(f"Deze kolommen worden al geschaald: {dubbel}.")
            case _:
                raise PrepPlanFout(f"Geen F4-stap: {stap!r}")

    # ---------- controleren ----------
    def vangnet(self, df: pd.DataFrame) -> list[str]:
        return [k for k in self.features(df) if _numeriek(df[k]) and not self._van(k, Imputeer)]

    def problemen(self, df: pd.DataFrame) -> list[str]:
        """Wat er nog moet gebeuren voordat een model met deze features kan werken (op basis van train)."""
        uit = []
        for k in self.features(df):
            if _numeriek(df[k]):
                if df[k].isna().any() and not self._van(k, Imputeer):
                    uit.append(f"'{k}' heeft {int(df[k].isna().sum())} NA in train en nog geen imputatie")
            elif not self._van(k, OneHot, TargetEncoding):
                uit.append(f"'{k}' is een categorie en heeft nog geen encoding (one-hot of target encoding)")
        if self.multi_imputatie() and sum(_numeriek(df[k]) for k in self.features(df)) < 2:
            uit.append("knn/iteratief imputeren heeft minstens twee numerieke kolommen nodig")
        return uit

    # ---------- bouwen ----------
    def bouw(self, df: pd.DataFrame) -> Pipeline:
        feats = self.features(df)
        laag_a = []
        for k in feats:
            stappen = []
            imp = next(iter(self._van(k, Imputeer)), None)
            if _numeriek(df[k]):
                if self._van(k, Afkappen):
                    stappen.append(("afkappen", IQRAfkapper()))
                if self._van(k, LogTransformatie):
                    stappen.append(("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
                if imp is None:
                    stappen.append(("vangnet", SimpleImputer(strategy="median")))
                elif imp.methode in EENVOUDIG:
                    stappen.append(("imputeer", SimpleImputer(strategy=EENVOUDIG[imp.methode], fill_value=imp.waarde if imp.waarde is not None else 0)))
                if s := self.geschaald(k, df):
                    stappen.append(("schaal", SCHALERS[s.methode]()))
            elif imp is not None:
                stappen.append(("imputeer", SimpleImputer(strategy=EENVOUDIG[imp.methode],
                                                          fill_value=str(imp.waarde) if imp.waarde is not None else "onbekend")))
            laag_a.append((k, Pipeline(stappen) if stappen else "passthrough", [k]))
        for s in self.stappen:
            if isinstance(s, NAIndicator):
                laag_a.append((f"{s.kolom}_ontbreekt", FunctionTransformer(is_na, feature_names_out=na_namen), [s.kolom]))
        lagen = [("per_kolom", ColumnTransformer(laag_a, remainder="drop", verbose_feature_names_out=False))]

        if m := self.multi_imputatie():
            imputer = KNNImputer(n_neighbors=5) if m == "knn" else IterativeImputer(random_state=0)
            lagen.append(("multi_imputatie", ColumnTransformer(
                [(m, imputer, make_column_selector(dtype_include=np.number))], remainder="passthrough", verbose_feature_names_out=False)))

        encoders = []
        for s in self.stappen:
            if isinstance(s, OneHot):
                encoders.append((f"{s.kolom}_onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [s.kolom]))
            elif isinstance(s, TargetEncoding):
                encoders.append((f"{s.kolom}_te", TargetEncoder(cv=KFold(5, shuffle=True, random_state=0)), [s.kolom]))
        if encoders:
            lagen.append(("encoders", ColumnTransformer(encoders, remainder="passthrough", verbose_feature_names_out=False)))
        return Pipeline(lagen).set_output(transform="pandas")

    def proef(self, df: pd.DataFrame) -> dict:
        """Fit op train en controleer dat het resultaat model-klaar is: alles numeriek, geen NA."""
        if p := self.problemen(df):
            raise PrepPlanFout("; ".join(p))
        d = df[df[self.target].notna()]
        uit = self.bouw(d).fit_transform(d[self.features(d)], d[self.target])
        niet_num = [k for k in uit.columns if not (pd.api.types.is_numeric_dtype(uit[k]) or pd.api.types.is_bool_dtype(uit[k]))]
        if niet_num or uit.isna().any().any():
            raise PrepPlanFout(f"Na de pipeline: niet-numeriek {niet_num}, NA in {list(uit.columns[uit.isna().any()])}")
        return {"rijen": len(uit), "kolommen": list(uit.columns), "vangnet": self.vangnet(df)}
