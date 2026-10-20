"""Versie 2: datacontract, API, rooktest en MLflow (geen netwerk)."""
import json

import pytest
from fastapi.testclient import TestClient

from dsharness.contract import Contract
from dsharness.resultaat import maak_resultaat
from dsharness.serve import maak_app, rooktest
from test_resultaat import verslag


@pytest.fixture(scope="module")
def map_(tmp_path_factory):
    m = tmp_path_factory.mktemp("resultaat")
    maak_resultaat(verslag(), m, uitvoeren=False)  # notebook wordt al in test_resultaat uitgevoerd
    return m


def test_bestanden(map_):
    for naam in ("model.joblib", "contract.json", "voorbeeld.json", "verslag.json", "model_card.md", "mlflow.db"):
        assert (map_ / naam).exists(), naam


def test_contract(map_):
    c = Contract.lees(map_ / "contract.json")
    rij = json.loads((map_ / "voorbeeld.json").read_text(encoding="utf-8"))[0]
    assert c.controleer([rij]) == ([], [])
    fouten, _ = c.controleer([{k: v for k, v in rij.items() if k != "glucose"}])
    assert fouten == ["rij 0: kolom 'glucose' ontbreekt"]
    fouten, _ = c.controleer([{**rij, "bmi": "dik"}])
    assert "moet een getal zijn" in fouten[0]
    _, w = c.controleer([{**rij, "geslacht": "X", "leeftijd": 200, "bmi": None, "extra": 1}])
    assert len(w) == 4


def test_api(map_):
    client = TestClient(maak_app(map_))
    assert client.get("/gezondheid").json() == {"status": "ok", "target": "diabetes", "taak": "classificatie"}
    rijen = client.get("/voorbeeld").json()["rijen"]
    r = client.post("/voorspel", json={"rijen": rijen})
    assert r.status_code == 200
    uit = r.json()
    assert len(uit["voorspellingen"]) == len(rijen) and set(uit["kansen"][0]) == {"ja", "nee"}
    assert abs(sum(uit["kansen"][0].values()) - 1) < 1e-9
    r = client.post("/voorspel", json={"rijen": [{**rijen[0], "bmi": None, "geslacht": "X"}]})
    assert r.status_code == 200 and len(r.json()["waarschuwingen"]) == 2
    assert client.post("/voorspel", json={"rijen": [{"bmi": 30}]}).status_code == 422
    assert "Model card" in client.get("/model").json()["model_card"]


def test_rooktest(map_):
    uit = rooktest(map_)
    assert uit["api_gelijk_aan_model"] and uit["lege_rij_geweigerd"] and uit["n"] == 3


def test_mlflow(map_):
    import mlflow
    from mlflow import MlflowClient

    from dsharness.tracking import tracking_uri
    mlflow.set_tracking_uri(tracking_uri(map_))
    client = MlflowClient(tracking_uri=tracking_uri(map_), registry_uri=tracking_uri(map_))
    exp = client.get_experiment_by_name("patient-test")
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 3  # ouder + baseline + lin
    gekozen = [r for r in runs if r.data.tags.get("gekozen") == "ja"]
    assert len(gekozen) == 1 and "test_roc_auc" in gekozen[0].data.metrics
    [versie] = client.search_model_versions("name='patient-test'")
    model = mlflow.sklearn.load_model(f"models:/patient-test/{versie.version}")
    rijen = json.loads((map_ / "voorbeeld.json").read_text(encoding="utf-8"))
    import pandas as pd
    assert len(model.predict(pd.DataFrame(rijen))) == 3
