import pytest

from src.model import ToxicityModel


@pytest.mark.unit
def test_produces_the_same_results():
    left = ToxicityModel()
    right = ToxicityModel()
    left.load()
    right.load()

    assert left.score("neutral text") == pytest.approx(right.score("neutral text"))
    assert left.score("you are an idiot") == pytest.approx(
        right.score("you are an idiot")
    )


@pytest.mark.unit
def test_keyword_detection_is_true_for_obvious_toxic_text():
    model = ToxicityModel()
    model.load()

    assert model.predict("you are stupid")
    assert not model.predict("thank you and have a nice day")
