"""DataLaag: de enige toegang tot de data. Na de split gaat de testset in de kluis.

- Vóór de split ziet iedereen alle data.
- Na de split geeft `zicht()` alleen nog de trainset.
- Vaste regels (VASTE_REGELS) werken op train én test: ze leren niets van de data.
- De testset komt er maar één keer uit, via `eindtest()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar

import numpy as np
import pandas as pd

from .acties import OnmogelijkNaarNA, Split, VerwijderDubbelen, VerwijderZonderTarget

T = TypeVar("T")


class KluisFout(RuntimeError):
    """Poging om de testset te gebruiken waar dat niet mag."""


@dataclass
class Verkenning:
    tool: str
    kolom: str | None
    gesplitst: bool


class DataLaag:
    def __init__(self, df: pd.DataFrame, *, target: str, seed: int = 42):
        if target not in df.columns:
            raise ValueError(f"Target '{target}' bestaat niet in de data.")
        self.target = target
        self.seed = seed
        self.log: list[Verkenning] = []
        self._df = df.reset_index(drop=True).copy()  # alleen via zicht()
        self._set: pd.Series | None = None
        self._test_gebruikt = False
        self.split_actie: Split | None = None

    def __repr__(self) -> str:
        return f"DataLaag(target={self.target!r}, gesplitst={self.gesplitst}, n_train={self.n_train}, n_test={self.n_test})"

    # ---------- lezen ----------
    @property
    def gesplitst(self) -> bool:
        return self._set is not None

    def zicht(self) -> pd.DataFrame:
        """Alle data vóór de split, alleen train erna. Altijd een kopie."""
        if self._set is None:
            return self._df.copy()
        return self._df.loc[self._set[self._set == "train"].index].copy()

    @property
    def n_train(self) -> int:
        return len(self.zicht())

    @property
    def n_test(self) -> int:
        return 0 if self._set is None else int((self._set == "test").sum())

    def noteer(self, tool: str, kolom: str | None = None) -> None:
        self.log.append(Verkenning(tool, kolom, self.gesplitst))

    def onderzocht(self, tools: tuple[str, ...], kolom: str | None = None) -> bool:
        """Is een van deze verkenningen gedaan (op deze kolom, of zonder kolom = alle kolommen)?"""
        return any(v.tool in tools and (v.kolom is None or kolom is None or v.kolom == kolom) for v in self.log)

    def dubbele_paren_over_sets(self) -> int:
        """Aantal dubbele rijen waarvan de eerdere kopie in een andere set zit (alleen voor de Reviewer)."""
        if self._set is None:
            return 0
        dup = self._df.duplicated(keep=False)
        n = 0
        for _, groep in self._df[dup].groupby(list(self._df.columns), dropna=False):
            if self._set.loc[groep.index].nunique() > 1:
                n += 1
        return n

    # ---------- uitvoeren ----------
    def voer_uit(self, actie) -> str:
        match actie:
            case VerwijderDubbelen():
                voor = len(self._df)
                self._houd(~self._df.duplicated())
                return f"{voor - len(self._df)} dubbele rij(en) verwijderd."
            case OnmogelijkNaarNA(kolom=k, regel=r):
                self._kolom(k)
                s = self._df[k]
                masker = {"<0": s < 0, "=0": s == 0, "<=0": s <= 0}[r]
                self._df.loc[masker, k] = np.nan
                return f"{int(masker.sum())} waarde(n) in '{k}' omgezet naar NA."
            case VerwijderZonderTarget():
                voor = len(self._df)
                self._houd(self._df[self.target].notna())
                return f"{voor - len(self._df)} rij(en) zonder target verwijderd."
            case Split():
                return self._split(actie)
        raise TypeError(f"Onbekende actie: {actie!r}")

    def _kolom(self, k: str) -> None:
        if k not in self._df.columns:
            raise ValueError(f"Kolom '{k}' bestaat niet.")

    def _houd(self, masker: pd.Series) -> None:
        self._df = self._df[masker]
        if self._set is not None:
            self._set = self._set.loc[self._df.index]

    def _split(self, a: Split) -> str:
        if self._set is not None:
            raise KluisFout("De data is al gesplitst.")
        rng = np.random.default_rng(self.seed)
        idx = self._df.index.to_numpy()
        test = np.zeros(len(idx), dtype=bool)
        if a.methode == "willekeurig":
            test[rng.permutation(len(idx))[: round(len(idx) * a.test_fractie)]] = True
        elif a.methode == "gestratificeerd":
            y = self._df[self.target].to_numpy()
            for k in pd.unique(y):
                pos = np.flatnonzero(y == k)
                test[rng.permutation(pos)[: round(len(pos) * a.test_fractie)]] = True
        elif a.methode == "tijd":
            if not a.tijd_kolom:
                raise ValueError("Bij een split op tijd is tijd_kolom verplicht.")
            self._kolom(a.tijd_kolom)
            volgorde = np.argsort(self._df[a.tijd_kolom].to_numpy(), kind="stable")
            test[volgorde[len(idx) - round(len(idx) * a.test_fractie):]] = True
        elif a.methode == "groep":
            if not a.groep_kolom:
                raise ValueError("Bij een split per groep is groep_kolom verplicht.")
            self._kolom(a.groep_kolom)
            g = self._df[a.groep_kolom].to_numpy()
            groepen = rng.permutation(pd.unique(g))
            test = np.isin(g, groepen[: round(len(groepen) * a.test_fractie)])
        self._set = pd.Series(np.where(test, "test", "train"), index=idx)
        self.split_actie = a
        return f"Gesplitst ({a.methode}): {self.n_train} rijen train, {self.n_test} rijen test. De testset staat in de kluis."

    # ---------- kluis ----------
    def eindtest(self, fn: Callable[[pd.DataFrame], T]) -> T:
        """Geeft de testset één keer vrij, aan `fn`. Daarna is hij 'verbrand'."""
        if self._set is None:
            raise KluisFout("Er is nog geen split.")
        if self._test_gebruikt:
            raise KluisFout("De testset is al gebruikt. Een nieuwe eindtest vraagt een nieuwe testset en akkoord van de PO.")
        self._test_gebruikt = True
        return fn(self._df.loc[self._set[self._set == "test"].index].copy())
