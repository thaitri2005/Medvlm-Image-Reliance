from medvlm.models.mock import MockVLM
from medvlm.probes import ABSTAIN_SUFFIX, opposite_partners, probe_config, probe_jobs, run_probes
from medvlm.run import done_keys, make_query
from medvlm.schema import Item

from .conftest import make_items


def _it(sid, img, q, ans, mod="ct", ds="slake"):
    return Item(sample_id=sid, dataset=ds, split="test", image_id=img, image_file=f"{img}.png", question=q,
                answer=ans, answer_type="yesno", modality=mod, region="", qgroup="presence", qtype_raw="x")


def test_opposite_partner_has_same_question_opposite_answer_other_image_same_modality():
    items = [_it("a", "i1", "Does the picture contain liver?", "yes"),
             _it("b", "i2", "does the picture contain liver", "No"),        # same after normalisation
             _it("c", "i3", "Does the picture contain liver?", "yes"),       # same answer: not a partner
             _it("d", "i4", "Does the picture contain liver?", "no", mod="mri"),   # other modality
             _it("e", "i1", "Does the picture contain liver?", "no")]        # same image: not a partner
    p = opposite_partners(items)
    assert p["a"].sample_id == "b"                                            # "e" shares a's image
    assert p["c"].sample_id in {"b", "e"}
    assert p["b"].sample_id in {"a", "c"}
    assert "d" not in p                                                       # no opposite answer in MRI


def test_probe_jobs_are_yesno_only_and_opp_uses_partner_image(paths):
    from PIL import Image

    items = make_items(paths)
    Image.new("RGB", (16, 16), (1, 2, 3)).save(paths.images / "slake" / "img9.png")
    items.append(_it("slake-test-x", "slake-9", "Is there a lesion in the left lung?", "no"))
    items[-1].image_file = "slake/img9.png"
    assert all(j.item.answer_type == "yesno" for p in ("blank", "abstain") for j in probe_jobs(p, items))
    opp = probe_jobs("opp", items)
    assert opp and all(j.image_file and j.image_file != j.item.image_file for j in opp)
    q = make_query(opp[0], paths, score_gold=False)
    assert q.image is not None and q.gold is None


def test_probe_config_isolates_runs_and_changes_only_the_instruction(cfg):
    c = probe_config(cfg, "abstain")
    assert c["model"]["name"].endswith("__abstain") and c["prompt"]["yesno_suffix"] == ABSTAIN_SUFFIX
    assert cfg["prompt"]["yesno_suffix"] != ABSTAIN_SUFFIX                    # original config untouched
    assert probe_config(cfg, "blank")["prompt"] == cfg["prompt"]


def test_run_probes_end_to_end_with_mock(paths, cfg):
    make_items(paths)
    run_probes(cfg, paths, MockVLM(cfg["prompt"]) if _takes_prompt() else MockVLM(), ["blank", "abstain"])
    for p in ("blank", "abstain"):
        keys = done_keys(paths.run_dir(f"{cfg['model']['name']}__{p}", cfg["profile"]))
        assert keys and all(k.endswith("|blank" if p == "blank" else "|noimg") for k in keys)


def _takes_prompt():
    import inspect

    return len(inspect.signature(MockVLM).parameters) > 0
