import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from dsharness import modellen
from dsharness.modellen import GeduldigModel, wachttijd


def test_wachttijd_uit_body():
    e = ModelHTTPError(429, "m", body={"error": {"message": "Please retry in 20.6s."}})
    assert wachttijd(e, 0) == pytest.approx(21.6)


def test_wachttijd_uit_header():
    e = ModelHTTPError(429, "m", body=None)
    e.headers = {"retry-after": "3"}
    assert wachttijd(e, 0) == 3


def test_opnieuw_bij_429(monkeypatch):
    async def geen_slaap(_):
        return None
    monkeypatch.setattr(modellen.asyncio, "sleep", geen_slaap)
    pogingen = []

    def f(messages, info):
        pogingen.append(1)
        if len(pogingen) < 3:
            raise ModelHTTPError(429, "m", body="retry in 1s")
        return ModelResponse(parts=[TextPart("hoi")])

    meldingen = []
    agent = Agent(GeduldigModel(FunctionModel(f), bij_wachten=meldingen.append))
    assert agent.run_sync("x").output == "hoi"
    assert len(pogingen) == 3 and len(meldingen) == 2


def test_geen_retry_bij_400():
    def f(messages, info):
        raise ModelHTTPError(400, "m", body="kapot")
    with pytest.raises(ModelHTTPError):
        Agent(GeduldigModel(FunctionModel(f))).run_sync("x")


def test_daglimiet_stopt_direct(monkeypatch):
    geslapen = []

    async def slaap(t):
        geslapen.append(t)
    monkeypatch.setattr(modellen.asyncio, "sleep", slaap)

    def f(messages, info):
        raise ModelHTTPError(429, "m", body={"details": [{"retryDelay": "65187s"}]})
    with pytest.raises(modellen.LimietBereikt, match="daglimiet"):
        Agent(GeduldigModel(FunctionModel(f))).run_sync("x")
    assert geslapen == []
