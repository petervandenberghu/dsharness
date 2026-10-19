import pytest

from dsharness.acties import OnmogelijkNaarNA, Split, VerwijderDubbelen, VerwijderZonderTarget
from dsharness.data import DataLaag, KluisFout
from dsharness.datasets import huizen, patient, winkel


def test_zicht_na_split_is_alleen_train():
    d = DataLaag(huizen().df, target="prijs_k")
    d.voer_uit(Split(methode="willekeurig", test_fractie=.25))
    assert d.n_train + d.n_test == 60 and d.n_test == 15
    assert len(d.zicht()) == 45


def test_vaste_regel_werkt_op_train_en_test():
    d = DataLaag(huizen().df, target="prijs_k")
    d.voer_uit(Split(methode="willekeurig"))
    d.voer_uit(OnmogelijkNaarNA(kolom="oppervlakte", regel="<0"))
    assert d.eindtest(lambda t: (t.oppervlakte < 0).sum()) == 0
    assert (d.zicht().oppervlakte < 0).sum() == 0


def test_eindtest_maar_een_keer():
    d = DataLaag(huizen().df, target="prijs_k")
    with pytest.raises(KluisFout):
        d.eindtest(len)
    d.voer_uit(Split(methode="willekeurig"))
    d.eindtest(len)
    with pytest.raises(KluisFout, match="al gebruikt"):
        d.eindtest(len)


def test_zicht_is_kopie():
    d = DataLaag(huizen().df, target="prijs_k")
    z = d.zicht()
    z.loc[0, "prijs_k"] = -1
    assert d.zicht().loc[0, "prijs_k"] != -1


def test_groep_split_houdt_patienten_bij_elkaar():
    d = DataLaag(patient().df, target="diabetes")
    d.voer_uit(Split(methode="groep", groep_kolom="patient_id"))
    train = set(d.zicht().patient_id)
    test = d.eindtest(lambda t: set(t.patient_id))
    assert train and test and not train & test


def test_tijd_split_test_is_toekomst():
    d = DataLaag(winkel().df, target="verkoop")
    d.voer_uit(Split(methode="tijd", tijd_kolom="datum"))
    assert d.zicht().datum.max() < d.eindtest(lambda t: t.datum.min())


def test_f2_regels():
    d = DataLaag(huizen().df, target="prijs_k")
    assert d.voer_uit(VerwijderDubbelen()).startswith("2 ")
    assert d.voer_uit(VerwijderZonderTarget()).startswith("1 ")
    assert d.voer_uit(OnmogelijkNaarNA(kolom="oppervlakte", regel="<0")).startswith("1 ")


def test_dubbele_paren_over_sets():
    d = DataLaag(huizen().df, target="prijs_k")
    d.voer_uit(Split(methode="willekeurig", test_fractie=.5))
    assert d.dubbele_paren_over_sets() in (0, 1, 2)


def test_tweede_split_kan_niet():
    d = DataLaag(huizen().df, target="prijs_k")
    d.voer_uit(Split(methode="willekeurig"))
    with pytest.raises(KluisFout):
        d.voer_uit(Split(methode="willekeurig"))


def test_geschiedenis():
    d = DataLaag(huizen().df, target="prijs_k")
    d.voer_uit(VerwijderDubbelen())
    d.voer_uit(Split(methode="willekeurig"))
    assert d.geschiedenis[0].startswith("Dubbele rijen verwijderen: 2 ") and "Train/test-split" in d.geschiedenis[1]
