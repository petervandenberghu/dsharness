"""Vaste bronnencatalogus. Agents citeren een sleutel, zodat ze geen URL's kunnen verzinnen."""
from __future__ import annotations

from typing import Literal

from .contracts import Bron

CATALOGUS: dict[str, tuple[str, str]] = {
    "crisp": ("Chapman et al. (2000), CRISP-DM 1.0", "https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining"),
    "pitfalls": ("scikit-learn: Common pitfalls (data leakage)", "https://scikit-learn.org/stable/common_pitfalls.html"),
    "impute": ("scikit-learn: Imputation of missing values", "https://scikit-learn.org/stable/modules/impute.html"),
    "cv": ("scikit-learn: Cross-validation (groepen en tijdreeksen)", "https://scikit-learn.org/stable/modules/cross_validation.html"),
    "tts": ("scikit-learn: train_test_split (stratify)", "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html"),
    "fimd_mech": ("Van Buuren (2018), FIMD §1.2: MCAR, MAR en MNAR", "https://stefvanbuuren.name/fimd/sec-MCAR.html"),
    "fimd_simple": ("Van Buuren (2018), FIMD §1.3: eenvoudige oplossingen", "https://stefvanbuuren.name/fimd/sec-simplesolutions.html"),
    "fes_missing": ("Kuhn & Johnson (2019), FES hfst. 8: missing data", "https://bookdown.org/max/FES/handling-missing-data.html"),
    "kapoor": ("Kapoor & Narayanan (2023), Leakage and the reproducibility crisis", "https://doi.org/10.1016/j.patter.2023.100804"),
    "kaufman": ("Kaufman et al. (2012), Leakage in data mining", "https://doi.org/10.1145/2382577.2382579"),
    "geron": ("Géron (2022), Hands-On ML hfst. 2: testset vroeg apart zetten", "https://github.com/ageron/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb"),
    "fpp": ("Hyndman & Athanasopoulos, FPP3 §5.10: tijdreeks-cross-validatie", "https://otexts.com/fpp3/tscv.html"),
    "tukey": ("IQR-regel voor uitschieters (Tukey's fences)", "https://en.wikipedia.org/wiki/Outlier#Tukey%27s_fences"),
}

BronSleutel = Literal[
    "crisp", "pitfalls", "impute", "cv", "tts", "fimd_mech", "fimd_simple",
    "fes_missing", "kapoor", "kaufman", "geron", "fpp", "tukey",
]


def bron(sleutel: str) -> Bron:
    titel, url = CATALOGUS[sleutel]
    return Bron(titel=titel, url=url)


def catalogus_tekst() -> str:
    return "\n".join(f"- {k}: {t}" for k, (t, _) in CATALOGUS.items())
