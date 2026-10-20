"""Lab: voert F5-acties uit. Cross-validatie op train, met de F4-pipeline binnen elke fold.

Preparatie, feature-selectie en het model worden per fold opnieuw gefit; de testset komt hier nooit in.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, make_scorer, recall_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit, cross_validate
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline

from .acties import CVOpzet, Experiment, KiesModel, omschrijf
from .data import DataLaag
from .prep_plan import PrepPlan

# model → (regressie, classificatie, toegestane parameters)
CATALOGUS = {
    "baseline": (lambda **p: DummyRegressor(strategy="mean"), lambda **p: DummyClassifier(strategy="prior"), set()),
    "lineair": (lambda **p: Ridge(**p), lambda **p: LogisticRegression(max_iter=2000, **p), {"alpha", "C"}),
    "random_forest": (lambda **p: RandomForestRegressor(random_state=0, **p), lambda **p: RandomForestClassifier(random_state=0, **p),
                      {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}),
    "gradient_boosting": (lambda **p: HistGradientBoostingRegressor(random_state=0, **p),
                          lambda **p: HistGradientBoostingClassifier(random_state=0, **p),
                          {"learning_rate", "max_depth", "max_iter", "min_samples_leaf", "l2_regularization"}),
    "knn": (lambda **p: KNeighborsRegressor(**p), lambda **p: KNeighborsClassifier(**p), {"n_neighbors", "weights"}),
}
ALLEEN = {"alpha": "regressie", "C": "classificatie"}
GEBALANCEERD_KAN = {"lineair", "random_forest", "gradient_boosting"}
HOGER_IS_BETER = {"rmse": False, "mae": False}


class LabFout(ValueError):
    """Een F5-actie kan (nog) niet."""


class Resultaat(BaseModel):
    naam: str
    omschrijving: str
    model: str
    metriek: str
    gemiddelde: float
    sd: float
    train: float
    folds: list[float]
    features: int


class Lab:
    def __init__(self, data: DataLaag, plan: PrepPlan, taak: str, positief: str | None = None):
        if not data.gesplitst:
            raise LabFout("F5 kan pas na de split.")
        self.data, self.plan, self.taak = data, plan, taak
        self.cv: CVOpzet | None = None
        self.resultaten: dict[str, Resultaat] = {}
        self.experimenten: dict[str, Experiment] = {}
        self.gekozen: str | None = None
        self.eind = None  # EindResultaat na F6
        self.eindmodel = None  # de pipeline, gefit op heel train
        y = data.zicht()[data.target]
        self.positief = positief if positief is not None else (str(y.value_counts().index[-1]) if taak == "classificatie" else None)

    # ---------- hulp ----------
    def beter(self, a: float, b: float) -> bool:
        return a > b if HOGER_IS_BETER.get(self.cv.metriek, True) else a < b

    def beste(self) -> Resultaat | None:
        if not self.resultaten:
            return None
        hoger = HOGER_IS_BETER.get(self.cv.metriek, True)
        return (max if hoger else min)(self.resultaten.values(), key=lambda r: r.gemiddelde)

    def gelijk_aan_baseline(self, r: Resultaat) -> bool:
        """Scoort dit model in elke fold hetzelfde als de baseline? Dan leert het niets (bijv. te weinig rijen om te splitsen)."""
        b = self.baseline()
        return b is not None and r.model != "baseline" and len(b.folds) == len(r.folds) and all(
            abs(x - y) < 1e-9 for x, y in zip(b.folds, r.folds))

    def baseline(self) -> Resultaat | None:
        return next((self.resultaten[n] for n, e in self.experimenten.items() if e.model == "baseline" and n in self.resultaten), None)

    def _scorer(self):
        m = self.cv.metriek
        if m == "rmse":
            return "neg_root_mean_squared_error"
        if m == "mae":
            return "neg_mean_absolute_error"
        if m == "f1":
            return make_scorer(f1_score, pos_label=self.positief, zero_division=0)
        if m == "recall":
            return make_scorer(recall_score, pos_label=self.positief, zero_division=0)
        return m  # r2, accuracy, balanced_accuracy, roc_auc

    def _splitter(self, df: pd.DataFrame):
        sc, k = self.cv.methode, self.cv.folds
        if sc == "groep":
            groepen = df[self.data.split_actie.groep_kolom]
            if self.taak == "classificatie":  # elke fold beide klassen, en geen groep in twee folds
                return StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0), groepen
            return GroupKFold(n_splits=k), groepen
        if sc == "tijd":
            return TimeSeriesSplit(n_splits=k), None
        if sc == "stratified":
            return StratifiedKFold(n_splits=k, shuffle=True, random_state=0), None
        return KFold(n_splits=k, shuffle=True, random_state=0), None

    def train_df(self) -> pd.DataFrame:
        df = self.data.zicht()
        df = df[df[self.data.target].notna()]
        if self.cv and self.cv.methode == "tijd" and (tk := getattr(self.data.split_actie, "tijd_kolom", None)):
            df = df.sort_values(tk, kind="stable")
        return df

    def pipeline(self, e: Experiment, df: pd.DataFrame) -> Pipeline:
        regressie, classificatie, toegestaan = CATALOGUS[e.model]
        p = {k: v for k, v in e.parameters.items() if v is not None}
        if e.gebalanceerd:
            p["class_weight"] = "balanced"
        model = (classificatie if self.taak == "classificatie" else regressie)(**p)
        stappen = [("prep", self.plan.bouw(df))]
        if e.selectie_k:
            stappen.append(("selectie", SelectKBest(f_classif if self.taak == "classificatie" else f_regression, k=e.selectie_k)))
        stappen.append(("model", model))
        return Pipeline(stappen)

    # ---------- controleren ----------
    def valideer(self, actie) -> None:
        match actie:
            case CVOpzet(metriek=m):
                if self.resultaten:
                    raise LabFout("Er zijn al experimenten gedraaid; de CV-opzet ligt vast.")
                regressie = m in ("rmse", "mae", "r2")
                if regressie != (self.taak == "regressie"):
                    raise LabFout(f"'{m}' past niet bij een {self.taak}-taak.")
            case Experiment(naam=n, model=mod, parameters=p, selectie_k=k, gebalanceerd=g):
                if self.cv is None:
                    raise LabFout("Leg eerst de CV-opzet vast (cv_opzet).")
                if n in self.experimenten:
                    raise LabFout(f"Er is al een experiment '{n}'.")
                if self.baseline() is None and mod != "baseline":
                    raise LabFout("Draai eerst een baseline; anders weet je niet of een model iets toevoegt.")
                toegestaan = CATALOGUS[mod][2]
                if fout := sorted(set(p) - toegestaan):
                    raise LabFout(f"Onbekende parameters voor {mod}: {fout}. Toegestaan: {sorted(toegestaan) or 'geen'}.")
                if fout := [x for x in p if ALLEEN.get(x, self.taak) != self.taak]:
                    raise LabFout(f"Parameter {fout} hoort niet bij {self.taak}.")
                if g and (self.taak != "classificatie" or mod not in GEBALANCEERD_KAN):
                    raise LabFout("Balanceren kan alleen bij classificatie met lineair, random_forest of gradient_boosting.")
                if k is not None and not 1 <= k < self.n_features():
                    raise LabFout(f"selectie_k moet tussen 1 en {self.n_features() - 1} liggen.")
            case KiesModel(experiment=n):
                if n not in self.resultaten:
                    raise LabFout(f"Experiment '{n}' bestaat niet of is niet gedraaid.")
                if len(self.resultaten) < 2:
                    raise LabFout("Vergelijk eerst minstens een baseline en één model.")
            case _:
                raise LabFout(f"Geen F5-actie: {actie!r}")

    def n_features(self) -> int:
        df = self.train_df()
        return len(self.plan.bouw(df).fit_transform(df[self.plan.features(df)], df[self.data.target]).columns)

    # ---------- uitvoeren ----------
    def voer_uit(self, actie) -> str:
        self.valideer(actie)
        match actie:
            case CVOpzet():
                self.cv = actie
                return f"Vastgelegd: {omschrijf(actie)}."
            case Experiment():
                r = self._draai(actie)
                self.experimenten[actie.naam] = actie
                self.resultaten[actie.naam] = r
                tekst = f"{omschrijf(actie)}: {r.metriek} = {r.gemiddelde:.3f} ± {r.sd:.3f} (train {r.train:.3f})."
                if self.gelijk_aan_baseline(r):
                    tekst += (" Let op: precies gelijk aan de baseline; dit model leert niets. Mogelijk zijn er te weinig "
                              "rijen per fold voor deze instellingen (bijv. min_samples_leaf).")
                return tekst
            case KiesModel(experiment=n):
                self.gekozen = n
                return f"Gekozen voor F6: '{n}'."
        raise AssertionError

    def _draai(self, e: Experiment) -> Resultaat:
        df = self.train_df()
        X, y = df[self.plan.features(df)], df[self.data.target]
        splitter, groepen = self._splitter(df)
        with warnings.catch_warnings(record=True) as gevangen:
            warnings.simplefilter("always")
            uit = cross_validate(self.pipeline(e, df), X, y, groups=groepen, cv=splitter, scoring=self._scorer(),
                                 return_train_score=True, error_score="raise")
        teken = -1 if self.cv.metriek in ("rmse", "mae") else 1
        val, tr = teken * uit["test_score"], teken * uit["train_score"]
        if np.isnan(val).any() or np.isnan(tr).any():
            reden = next((str(w.message) for w in gevangen if "score" in str(w.message).lower()), "onbekend")
            raise LabFout(f"De {self.cv.metriek} kon in sommige folds niet berekend worden ({reden[:200]}). "
                          "Kies minder folds of een andere maat.")
        return Resultaat(naam=e.naam, omschrijving=omschrijf(e), model=e.model, metriek=self.cv.metriek,
                         gemiddelde=float(np.mean(val)), sd=float(np.std(val)), train=float(np.mean(tr)),
                         folds=[round(float(v), 4) for v in val], features=self.n_features() if e.selectie_k is None else e.selectie_k)

    def gekozen_pipeline(self) -> Pipeline:
        if self.gekozen is None:
            raise LabFout("Er is nog geen model gekozen.")
        return self.pipeline(self.experimenten[self.gekozen], self.train_df())

    def overzicht(self) -> str:
        if self.cv is None:
            return "Nog geen CV-opzet."
        regels = [f"CV: {omschrijf(self.cv)}" + (f"; positieve klasse '{self.positief}'" if self.positief else "")]
        for r in sorted(self.resultaten.values(), key=lambda r: r.gemiddelde, reverse=HOGER_IS_BETER.get(self.cv.metriek, True)):
            let_op = " [GELIJK AAN BASELINE: leert niets]" if self.gelijk_aan_baseline(r) else ""
            regels.append(f"- {r.naam}: {r.metriek} {r.gemiddelde:.3f} ± {r.sd:.3f}, train {r.train:.3f} ({r.omschrijving}){let_op}")
        if self.gekozen:
            regels.append(f"Gekozen: {self.gekozen}")
        return "\n".join(regels)


# ---------- F6: eenmalige eindtest ----------
class EindResultaat(BaseModel):
    experiment: str
    metriek: str
    test: float
    test_baseline: float | None
    cv_gemiddelde: float
    cv_sd: float
    n_train: int
    n_test: int
    extra: dict[str, float]
    binnen_verwachting: bool


EXTRA = {"regressie": ("rmse", "mae", "r2"), "classificatie": ("accuracy", "balanced_accuracy", "f1", "recall", "roc_auc")}


def _score(lab: Lab, metriek: str, model, X, y) -> float:
    from sklearn.metrics import get_scorer
    oud, lab.cv = lab.cv, lab.cv.model_copy(update={"metriek": metriek})
    try:
        s = lab._scorer()
        waarde = (get_scorer(s) if isinstance(s, str) else s)(model, X, y)
    finally:
        lab.cv = oud
    return float(-waarde if metriek in ("rmse", "mae") else waarde)


def voer_eindtest(lab: Lab, experiment: str) -> EindResultaat:
    """Fit op heel train, beoordeel één keer op de testset (uit de kluis). Daarna is de testset verbrand."""
    if lab.gekozen != experiment:
        raise LabFout(f"Alleen het gekozen model ('{lab.gekozen}') mag de eindtest doen.")
    df = lab.train_df()
    X, y = df[lab.plan.features(df)], df[lab.data.target]
    model = lab.gekozen_pipeline().fit(X, y)
    base_exp = next((e for e in lab.experimenten.values() if e.model == "baseline"), None)
    baseline = lab.pipeline(base_exp, df).fit(X, y) if base_exp else None

    def beoordeel(test: pd.DataFrame) -> dict:
        test = test[test[lab.data.target].notna()]
        Xt, yt = test[lab.plan.features(df)], test[lab.data.target]
        uit = {"n": len(test), "test": _score(lab, lab.cv.metriek, model, Xt, yt),
               "baseline": _score(lab, lab.cv.metriek, baseline, Xt, yt) if baseline is not None else None, "extra": {}}
        for m in EXTRA[lab.taak]:
            try:
                uit["extra"][m] = round(_score(lab, m, model, Xt, yt), 4)
            except (ValueError, AttributeError):  # bijv. roc_auc zonder beide klassen in test
                pass
        return uit

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = lab.data.eindtest(beoordeel)
    cv = lab.resultaten[experiment]
    lab.eindmodel = model
    lab.eind = EindResultaat(experiment=experiment, metriek=lab.cv.metriek, test=r["test"], test_baseline=r["baseline"],
                             cv_gemiddelde=cv.gemiddelde, cv_sd=cv.sd, n_train=len(df), n_test=r["n"], extra=r["extra"],
                             binnen_verwachting=abs(r["test"] - cv.gemiddelde) <= 2 * cv.sd + 1e-9)
    return lab.eind
