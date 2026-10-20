"""MLflow: experimenten (F5), eindtest (F6) en het model (registry), lokaal in de resultaatmap.

Bekijken:  uv run mlflow ui --backend-store-uri sqlite:///resultaat/mlflow.db
"""
from __future__ import annotations

import os
from pathlib import Path

from .acties import omschrijf
from .verslag import Reproductie, Verslag

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def tracking_uri(map: Path) -> str:
    return f"sqlite:///{(map / 'mlflow.db').resolve().as_posix()}"


def log_naar_mlflow(v: Verslag, r: Reproductie, map: str | Path) -> dict:
    import mlflow
    import mlflow.sklearn
    import sklearn
    from mlflow.models import infer_signature

    from .resultaat import model_card

    map = Path(map)
    uri = tracking_uri(map)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    exp = mlflow.get_experiment_by_name(v.project)
    exp_id = exp.experiment_id if exp else mlflow.create_experiment(
        v.project, artifact_location=(map / "mlartifacts").resolve().as_uri())
    df = r.lab.train_df()
    X = df[r.plan.features(df)]
    with mlflow.start_run(experiment_id=exp_id, run_name=f"{v.project} {v.gemaakt:%Y-%m-%d %H:%M}") as ouder:
        mlflow.log_params({"target": v.target, "taak": v.taak, "split": r.data.split_actie.methode if r.data.split_actie else "-",
                           "cv": f"{v.cv.methode}, {v.cv.folds} folds" if v.cv else "-", "metriek": v.cv.metriek if v.cv else "-",
                           "gekozen": v.gekozen or "-", "n_train": r.data.n_train, "n_test": r.data.n_test})
        mlflow.set_tags({"data_policy": v.data_policy, "maker": "dsharness",
                         "preparatie": " | ".join(omschrijf(s) for s in v.prep)[:5000]})
        mlflow.log_text(model_card(v), "model_card.md")
        mlflow.log_text(v.model_dump_json(indent=1), "verslag.json")
        model_run = None
        for res in r.lab.resultaten.values():
            e = r.lab.experimenten[res.naam]
            with mlflow.start_run(experiment_id=exp_id, run_name=res.naam, nested=True) as kind:
                mlflow.log_params({"model": e.model, **{f"p_{k}": v_ for k, v_ in e.parameters.items()},
                                   "selectie_k": e.selectie_k, "gebalanceerd": e.gebalanceerd})
                mlflow.log_metrics({f"cv_{res.metriek}": res.gemiddelde, f"cv_{res.metriek}_sd": res.sd, f"train_{res.metriek}": res.train})
                for i, f in enumerate(res.folds):
                    mlflow.log_metric(f"fold_{res.metriek}", f, step=i)
                if res.naam == v.gekozen and r.lab.eind is not None:
                    eind = r.lab.eind
                    mlflow.log_metrics({f"test_{eind.metriek}": eind.test, **{f"test_{k}": x for k, x in eind.extra.items()}})
                    if eind.test_baseline is not None:
                        mlflow.log_metric(f"test_baseline_{eind.metriek}", eind.test_baseline)
                    voorbeeld = X.head(3)
                    mlflow.sklearn.log_model(r.lab.eindmodel, name="model", signature=infer_signature(voorbeeld, r.lab.eindmodel.predict(voorbeeld)),
                                             input_example=voorbeeld, registered_model_name=v.project,
                                             serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                                             pip_requirements=[f"scikit-learn=={sklearn.__version__}", "pandas", "dsharness"])
                    mlflow.set_tag("gekozen", "ja")
                    model_run = kind.info.run_id
    return {"tracking_uri": uri, "run_id": ouder.info.run_id, "model_run_id": model_run}
