import pytest
from pydantic import ValidationError

from dsharness.config import Config, laad

BASIS = dict(project="p", doel="d", target="prijs_k", taak="regressie", data="oefenset:huizen")


def test_local_only_weigert_cloudmodel():
    with pytest.raises(ValidationError, match="cloudmodel"):
        Config(**BASIS, data_policy="local_only", modellen={"data_scout": "ollama:gemma4:12b", "reviewer": "google:gemini-flash-latest", "verkenner": None, "prep_engineer": None, "modeler": None})


def test_local_only_met_lokale_modellen():
    c = Config(**BASIS, data_policy="local_only", modellen={"data_scout": "ollama:gemma4:12b", "reviewer": None, "verkenner": "ollama:gemma4:12b", "prep_engineer": None, "modeler": None})
    assert c.model("reviewer") is None


def test_rol_ontbreekt():
    with pytest.raises(ValidationError, match="reviewer"):
        Config(**BASIS, data_policy="cloud_allowed", modellen={"data_scout": "ollama:gemma4:12b"})


def test_project_yaml_laadt(tmp_path):
    cfg, basis = laad("project.yaml")
    assert cfg.laad_data(basis).shape == (60, 6)


def test_lijst_modellen_en_policy():
    with pytest.raises(ValidationError, match="cloudmodel"):
        Config(**BASIS, data_policy="local_only",
               modellen={"data_scout": ["ollama:gemma4:12b", "google:gemini-flash-latest"], "reviewer": None, "verkenner": None, "prep_engineer": None, "modeler": None})
    c = Config(**BASIS, data_policy="cloud_allowed",
               modellen={"data_scout": ["google:gemini-flash-latest", "ollama:gemma4:12b"], "reviewer": None, "verkenner": None, "prep_engineer": None, "modeler": None})
    assert c.model("data_scout")[1] == "ollama:gemma4:12b"
