"""F3: Verkenner op de trainset, met een gescript model (geen netwerk)."""
import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from dsharness.data import DataLaag, KluisFout
from dsharness.datasets import huizen, klant
from dsharness.po import ScriptedPO
from dsharness.reviewer import Reviewer
from dsharness.scout import maak_scout
from dsharness.tools import AgentDeps, train_tools
from dsharness.verkenner import F3Rapport, KolomProfiel, controle, maak_verkenner
from dsharness.workflow import Team, verken_train


def na_f1(oefenset):
    o = oefenset()
    d = DataLaag(o.df, target=o.target)
    for a in o.juiste_f1:
        d.voer_uit(a)
    return o, d


def profielen(d: DataLaag, mechanisme: dict[str, str]) -> list[dict]:
    df = d.zicht()
    uit = []
    for k in df.columns:
        if k == d.target:
            continue
        n = int(df[k].isna().sum())
        uit.append({"kolom": k, "type": "getal" if df[k].dtype.kind in "fi" else "categorie", "na_aantal": n,
                    "na_mechanisme": mechanisme.get(k, "MCAR") if n else "geen_na", "verdeling": "v"})
    return uit


def script(d: DataLaag, mechanisme: dict[str, str]):
    na_kolommen = [k for k in d.zicht().columns if d.zicht()[k].isna().any()]

    def f(messages, info: AgentInfo) -> ModelResponse:
        delen = [p for m in messages for p in getattr(m, "parts", [])]
        uit = info.output_tools[0].name
        if not any(isinstance(p, ToolReturnPart) for p in delen):
            calls = [ToolCallPart("na_telling", {}), ToolCallPart("correlatie", {})]
            for k in na_kolommen:
                calls += [ToolCallPart("na_patroon", {"kolom": k}), ToolCallPart("vraag_po", {"vraag": f"Waarom ontbreekt {k}?"})]
            return ModelResponse(parts=calls)
        p = profielen(d, mechanisme)
        if not any(isinstance(x, RetryPromptPart) for x in delen):
            p = p[1:]  # eerste keer: één profiel vergeten
        return ModelResponse(parts=[ToolCallPart(uit, {"samenvatting": "s", "target": "t", "profielen": p, "voor_f4": ["x"]})])
    return FunctionModel(f)


def test_verkenner_vult_aan_en_verandert_niets():
    o, d = na_f1(klant)
    voor = d.zicht()
    po = ScriptedPO(o.po_kennis)
    team = Team(scout=maak_scout(None), reviewer=Reviewer(), verkenner=maak_verkenner(script(d, o.na_mechanismen)))
    r = asyncio.run(verken_train(po, d, doel=o.doel, taak=o.taak, team=team))
    assert {p.kolom for p in r.profielen} == set(voor.columns) - {o.target}
    assert {p.kolom: p.na_mechanisme for p in r.profielen if p.na_aantal} == o.na_mechanismen
    assert any("hoog inkomen" in a for _, a in po.vragen)
    assert d.zicht().equals(voor) and po.rapporten == [r]


def test_f3_vereist_split():
    o = klant()
    d = DataLaag(o.df, target=o.target)
    team = Team(scout=maak_scout(None), reviewer=Reviewer(), verkenner=maak_verkenner(None))
    with pytest.raises(KluisFout):
        asyncio.run(verken_train(ScriptedPO(), d, doel=o.doel, taak=o.taak, team=team))


def test_train_tools_weigeren_voor_de_split():
    o = klant()
    d = DataLaag(o.df, target=o.target)

    def f(messages, info):
        retries = [p for m in messages for p in getattr(m, "parts", []) if isinstance(p, RetryPromptPart)]
        if retries:
            return ModelResponse(parts=[TextPart(str(retries[0].content))])
        return ModelResponse(parts=[ToolCallPart("na_patroon", {"kolom": "inkomen_k"})])

    agent = Agent(FunctionModel(f), deps_type=AgentDeps, toolsets=[train_tools()])
    uit = agent.run_sync("x", deps=AgentDeps(data=d, po=ScriptedPO(), doel="", taak=o.taak)).output
    assert "pas na de split" in uit and not d.log


def _rapport(d, **wijzig):
    p = profielen(d, {"kamers": "MCAR", "bouwjaar": "MAR", "oppervlakte": "MCAR"})
    for x in p:
        x.update(wijzig.get(x["kolom"], {}))
    return F3Rapport(samenvatting="s", target="t", profielen=[KolomProfiel(**x) for x in p], voor_f4=[])


def test_controle_eist_onderzoek_en_po():
    o, d = na_f1(huizen)
    deps = AgentDeps(data=d, po=ScriptedPO(), doel="", taak=o.taak)
    mist = " ".join(controle(_rapport(d), d, deps))
    assert "'bouwjaar': draai eerst na_patroon" in mist and "'bouwjaar': vraag de PO" in mist
    for k in ("kamers", "bouwjaar", "oppervlakte"):
        d.noteer("na_patroon", k)
        deps.po_vragen.append((f"Waarom ontbreekt {k}?", "weet ik niet"))
    assert controle(_rapport(d), d, deps) == []


def test_controle_vangt_verkeerde_getallen():
    o, d = na_f1(huizen)
    deps = AgentDeps(data=d, po=ScriptedPO(), doel="", taak=o.taak)
    mist = " ".join(controle(_rapport(d, kamers={"na_aantal": 9}, wijk={"na_mechanisme": "MCAR"}), d, deps))
    assert "'kamers' heeft 2 NA" in mist and "'wijk' heeft geen NA" in mist
