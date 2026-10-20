"""API rond een resultaatmap (F7): model.joblib + contract.json + model_card.md.

Start:  uv run dsharness serve resultaat/ --port 8000   (docs op http://localhost:8000/docs)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .contract import Contract


class Invoer(BaseModel):
    rijen: list[dict[str, Any]] = Field(min_length=1, max_length=10_000)


class Uitvoer(BaseModel):
    voorspellingen: list[Any]
    kansen: list[dict[str, float]] | None = None
    waarschuwingen: list[str] = []


def maak_app(map: str | Path) -> FastAPI:
    map = Path(map)
    model = joblib.load(map / "model.joblib")
    contract = Contract.lees(map / "contract.json")
    card = (map / "model_card.md").read_text(encoding="utf-8") if (map / "model_card.md").exists() else ""
    voorbeeld = (map / "voorbeeld.json").read_text(encoding="utf-8") if (map / "voorbeeld.json").exists() else "[]"
    app = FastAPI(title=f"dsharness · {contract.target}", description="Voorspellingen van een model gemaakt met dsharness.")

    @app.get("/gezondheid")
    def gezondheid() -> dict:
        return {"status": "ok", "target": contract.target, "taak": contract.taak}

    @app.get("/model")
    def model_info() -> dict:
        return {"model_card": card, "contract": contract.model_dump()}

    @app.get("/voorbeeld")
    def voorbeeld_invoer() -> dict:
        import json
        return {"rijen": json.loads(voorbeeld)}

    @app.post("/voorspel", response_model=Uitvoer)
    def voorspel(invoer: Invoer) -> Uitvoer:
        fouten, waarschuwingen = contract.controleer(invoer.rijen)
        if fouten:
            raise HTTPException(status_code=422, detail={"fouten": fouten, "waarschuwingen": waarschuwingen})
        X = pd.DataFrame(invoer.rijen)[[k.naam for k in contract.kolommen]]
        for k in contract.kolommen:  # JSON kent geen NaN; zet ontbrekend om naar het juiste type
            X[k.naam] = pd.to_numeric(X[k.naam], errors="coerce") if k.type == "getal" else X[k.naam].astype(object)
        uit = Uitvoer(voorspellingen=[v.item() if hasattr(v, "item") else v for v in model.predict(X)], waarschuwingen=waarschuwingen)
        if contract.taak == "classificatie" and hasattr(model, "predict_proba"):
            klassen = [str(k) for k in model.classes_]
            uit.kansen = [{k: float(p) for k, p in zip(klassen, rij)} for rij in model.predict_proba(X)]
        return uit

    return app


def rooktest(map: str | Path) -> dict:
    """Deploy-check: start de API in-process, voorspel op de voorbeeldrijen en vergelijk met het model zelf."""
    import json

    from fastapi.testclient import TestClient

    map = Path(map)
    client = TestClient(maak_app(map))
    rijen = json.loads((map / "voorbeeld.json").read_text(encoding="utf-8"))
    assert client.get("/gezondheid").json()["status"] == "ok"
    r = client.post("/voorspel", json={"rijen": rijen})
    assert r.status_code == 200, r.text
    contract = Contract.lees(map / "contract.json")
    X = pd.DataFrame(rijen)[[k.naam for k in contract.kolommen]]
    verwacht = [v.item() if hasattr(v, "item") else v for v in joblib.load(map / "model.joblib").predict(X)]
    api = r.json()["voorspellingen"]
    gelijk = all((abs(a - b) < 1e-9) if isinstance(a, float) else a == b for a, b in zip(api, verwacht))
    fout = client.post("/voorspel", json={"rijen": [{}]})
    return {"api_gelijk_aan_model": gelijk, "n": len(rijen), "voorspellingen": api,
            "lege_rij_geweigerd": fout.status_code == 422}
