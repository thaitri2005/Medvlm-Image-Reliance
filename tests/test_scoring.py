import pytest

from medvlm.scoring import is_correct, normalize, parse_yesno, same_answer, token_f1


def test_normalize_strips_case_punctuation_articles():
    assert normalize("The Left-Lung.") == "left lung"


@pytest.mark.parametrize("pred,gold,expected", [
    ("lung", "Lung", 1.0), ("left lung", "lung", pytest.approx(2 / 3)),
    ("brain", "lung", 0.0), ("", "lung", 0.0), ("", "", 1.0)])
def test_token_f1(pred, gold, expected):
    assert token_f1(pred, gold) == expected


@pytest.mark.parametrize("text,label", [("Yes.", "yes"), ("no", "no"), ("No, there is not", "no"),
                                        ("I think yes", "yes"), ("unclear", None)])
def test_parse_yesno(text, label):
    assert parse_yesno(text) == label


def test_correctness_by_kind():
    assert is_correct("yesno", "Yes", "yes")
    assert not is_correct("yesno", "no", "yes")
    assert not is_correct("yesno", "maybe", "yes")
    assert is_correct("open", "the lung", "Lung")
    assert not is_correct("open", "liver", "Lung")


def test_same_answer():
    assert same_answer("yesno", "Yes.", "yes")
    assert not same_answer("yesno", "yes", "no")
    assert same_answer("open", "left lung", "lung", f1_thresh=0.5)
    assert not same_answer("open", "liver", "lung")
