import pytest
from pydantic import ValidationError

from dsharness.beslislog import Beslislog
from dsharness.contracts import Besluit, Voorstel
from dsharness.voorbeeld import voorbeeld_voorstel


def test_akkoord_bij_fout_kan_niet():
    with pytest.raises(ValidationError, match="alleen overrulen"):
        Besluit(voorstel_id="x", keuze="akkoord", review_oordeel="fout")


def test_overrulen_alleen_bij_fout():
    with pytest.raises(ValidationError, match="alleen bij een 'fout'"):
        Besluit(voorstel_id="x", keuze="overrulen", motivatie="want", review_oordeel="twijfel")


@pytest.mark.parametrize("keuze", ["aanpassen", "overrulen"])
def test_motivatie_verplicht(keuze):
    with pytest.raises(ValidationError, match="motivatie verplicht"):
        Besluit(voorstel_id="x", keuze=keuze, motivatie="  ", review_oordeel="fout")


def test_afwijzen_zonder_motivatie_mag():
    assert Besluit(voorstel_id="x", keuze="afwijzen", review_oordeel="goed").keuze == "afwijzen"


def test_voorstel_zonder_bron_kan_niet():
    d = voorbeeld_voorstel().model_dump()
    d["bronnen"] = []
    with pytest.raises(ValidationError):
        Voorstel.model_validate(d)


def test_beslislog_heen_en_terug(tmp_path):
    log = Beslislog(tmp_path / "log.jsonl")
    v = voorbeeld_voorstel("fout")
    b = Besluit(voorstel_id=v.id, keuze="overrulen", motivatie="PO weet dat test hier niet meetelt", review_oordeel="fout")
    log.schrijf(v, b)
    [(v2, b2)] = log.lees()
    assert v2 == v and b2.keuze == "overrulen" and b2.motivatie == b.motivatie
