"""Integration test of the REAL Hugging Face code path (needs network + a model download).

    pytest -m slow                                   # SmolVLM-256M on CPU (laptop)
    MEDVLM_TEST_MODEL=medgemma4b pytest -m slow      # first thing to run on Colab / the big machine

Each speed-up in hf_vlm.py has an equivalence test against the plain path; padding is forced to
happen (batch_size is set explicitly, not taken from the config).
"""
import math

import pytest
from PIL import Image

from medvlm.models.base import Query

pytestmark = pytest.mark.slow
import os  # noqa: E402


@pytest.fixture(scope="module")
def vlm():
    from medvlm.config import load_config
    from medvlm.models import load_model

    return load_model(load_config("smoke", os.environ.get("MEDVLM_TEST_MODEL", "smolvlm")))


def img(color):
    return Image.new("RGB", (96, 96), color)


def _oom_skip(fn):
    """Small GPUs may not hold a batch of 3 images; that is a skip, not a failure."""
    try:
        return fn()
    except Exception as e:   # noqa: BLE001
        if "out of memory" in str(e).lower():
            pytest.skip("GPU too small for a batch of 3 in this test; lower-batch equivalence covered elsewhere")
        raise


def test_yesno_logits_with_and_without_image(vlm):
    outs = vlm.generate([Query("Is there a lesion?", "yesno", img("red"), "yes"),
                         Query("Is there a lesion?", "yesno", None, "yes")])
    for o in outs:
        assert o.text in ("yes", "no")
        assert 0.0 <= o.p_yes <= 1.0 and o.gold_logp <= 0.0 and o.entropy >= 0.0
        assert 0.0 < o.yn_mass <= 1.0 + 1e-4
        assert math.isclose(o.gold_logp, math.log(o.p_yes), rel_tol=1e-3, abs_tol=1e-4)


def test_open_generation_and_teacher_forced_gold(vlm):
    outs = vlm.generate([Query("What organ is shown?", "open", img("blue"), "lung"),
                         Query("What organ is shown?", "open", None, "lung")])
    for o in outs:
        assert isinstance(o.text, str) and o.text
        assert o.gold_logp is not None and o.gold_logp < 0.0
        assert o.mean_logprob is not None and o.mean_logprob <= 0.0


def test_batching_with_left_padding_matches_single_queries(vlm):
    """Different prompt lengths force padding; padding must not change the answer."""
    qs = [Query("Is there a lesion?", "yesno", img("red")),
          Query("Is there a very small lesion visible in the upper left part of the lung field?", "yesno", img("green")),
          Query("Any?", "yesno", img("blue"))]
    old, vlm.batch_size = vlm.batch_size, 3
    try:
        batched = _oom_skip(lambda: vlm.generate(qs))
        vlm.batch_size = 1
        singles = [vlm.generate([q])[0] for q in qs]
    finally:
        vlm.batch_size = old
    for s, b in zip(singles, batched):
        assert abs(s.p_yes - b.p_yes) < 0.05


def test_batched_gold_scoring_matches_one_at_a_time(vlm):
    """Gold answers of different token lengths, left-padded in one batch."""
    qs = [Query("What organ is shown?", "open", img("red"), "lung"),
          Query("Where is the abnormality located in this image?", "open", img("green"), "lower left lung"),
          Query("Which plane?", "open", img("blue"), "transverse")]
    batched = _oom_skip(lambda: vlm._gold_logp(qs))
    singles = [vlm._gold_logp([q])[0] for q in qs]
    for s, b in zip(singles, batched):
        assert abs(s - b) < 0.15


def test_logits_to_keep_gives_the_same_answers_as_full_logits(vlm):
    q = [Query("Is there a lesion?", "yesno", img("red"), "yes")]
    qo = [Query("What organ is shown?", "open", img("red"), "lung")]
    fast_y, fast_g = vlm.generate(q)[0], vlm._gold_logp(qo)[0]
    vlm.use_ltk = False
    try:
        slow_y, slow_g = vlm.generate(q)[0], vlm._gold_logp(qo)[0]
    finally:
        vlm.use_ltk = True
    assert abs(fast_y.p_yes - slow_y.p_yes) < 1e-3 and abs(fast_g - slow_g) < 1e-2


def test_removing_the_image_changes_the_input(vlm):
    q = "Is there a lesion?"
    with_img = vlm.generate([Query(q, "yesno", img("red"), "yes")])[0]
    without = vlm.generate([Query(q, "yesno", None, "yes")])[0]
    assert with_img.p_yes != without.p_yes        # image tokens are really present / absent
