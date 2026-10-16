"""Opposite-answer (contrast-set) probe R4: manifest filter, runner, outcome cells, registered H12 family."""
import pandas as pd
from PIL import Image

from medvlm.analysis.contamination import train_overlap
from medvlm.analysis.contrast import summarize_contrast
from medvlm.data.sampling import keep_with_opposite_partner
from medvlm.models.base import VLM, Output
from medvlm.probes import opposite_partners, run_contrast
from medvlm.schema import Item, YESNO, write_jsonl

# question -> answer on each of 8 images; every image has an opposite-answer partner for its question
TRUTH = {"is liver present?": {0: "yes", 1: "no", 2: "yes", 3: "no"},
         "is bone present?": {4: "no", 5: "yes", 6: "no", 7: "yes"}}
SETS = (("r4_pathvqa", "contrast", None, "PathVQA, probe R4"),)


def _color(i):
    return (20 + 25 * i, 200 - 20 * i, 90)


def make_contrast(paths):
    items = []
    (paths.images / "pathvqa").mkdir(parents=True, exist_ok=True)
    for q, answers in TRUTH.items():
        for i, a in answers.items():
            rel = f"pathvqa/img{i}.png"
            Image.new("RGB", (16, 16), _color(i)).save(paths.images / rel)
            items.append(Item(sample_id=f"pathvqa-test-{i}", dataset="pathvqa", split="test", image_id=f"p{i}",
                              image_file=rel, question=q, answer=a, answer_type=YESNO, modality="pathology",
                              region="", qgroup="presence", qtype_raw="x"))
    write_jsonl(paths.manifest_dir("contrast") / "items.jsonl", (it.to_dict() for it in items))
    return items


class Oracle(VLM):
    """Reads the image: answers correctly for whichever image it is shown."""
    name = "oracle"

    def generate(self, queries):
        by_color = {_color(i): (q, a) for q, ans in TRUTH.items() for i, a in ans.items()}
        outs = []
        for q in queries:
            _, a = by_color[q.image.getpixel((0, 0))]
            p = 0.9 if a == "yes" else 0.1
            outs.append(Output("yes" if p >= .5 else "no", p_yes=p, yn_mass=1.0))
        return outs


class Blind(VLM):
    """Ignores the image: always 'yes'."""
    name = "blind"

    def generate(self, queries):
        return [Output("yes", p_yes=0.8, yn_mass=1.0) for _ in queries]


def _run(paths, models: dict):
    run_contrast(list(models), [], None, loader=lambda cfg: models[cfg["model"]["name"]](), paths=paths)


def test_filter_keeps_only_yesno_questions_with_an_opposite_answer_on_another_image():
    rows = [dict(sample_id="a", dataset="d", modality="m", image_id="1", question="Is X present?", answer="yes", answer_type="yesno"),
            dict(sample_id="b", dataset="d", modality="m", image_id="2", question="is x present", answer="No", answer_type="yesno"),
            dict(sample_id="c", dataset="d", modality="m", image_id="3", question="Is Y present?", answer="yes", answer_type="yesno"),
            dict(sample_id="d", dataset="d", modality="m", image_id="3", question="Is Y present?", answer="no", answer_type="yesno"),
            dict(sample_id="e", dataset="d", modality="m", image_id="4", question="What is X?", answer="liver", answer_type="open")]
    assert [r["sample_id"] for r in keep_with_opposite_partner(rows)] == ["a", "b"]   # same-image partner does not count


def test_partners_are_opposite_and_on_another_image(paths):
    items = make_contrast(paths)
    p = opposite_partners(items)
    assert len(p) == len(items)
    assert all(p[i.sample_id].image_id != i.image_id and p[i.sample_id].answer != i.answer for i in items)


def test_oracle_follows_and_scores_pair_accuracy_one_blind_model_scores_zero(paths):
    make_contrast(paths)
    _run(paths, {"medgemma4b": Oracle, "gemma3_4b": Blind})
    out = summarize_contrast(paths, models=("medgemma4b", "gemma3_4b"), pairs=(("medgemma4b", "gemma3_4b"),),
                             sets=SETS, B=50, seed=0)
    m = out["contrast_models"].set_index("model")
    assert m.loc["medgemma4b", "follow"] == 1 and m.loc["medgemma4b", "both_right"] == 1
    assert m.loc["gemma3_4b", "follow"] == 0 and m.loc["gemma3_4b", "both_right"] == 0   # 0 by construction
    assert m.loc["gemma3_4b", "stuck"] == 1
    d = out["contrast_pairs"].iloc[0]
    assert d.d_follow == 1 and d.d_both_right == 1
    assert out["contrast_tests"].empty                     # the registered family needs all three pairs


def test_registered_h12_family_is_tested_only_when_complete_and_holm_adjusted(paths):
    make_contrast(paths)
    _run(paths, {"medgemma4b": Oracle, "medgemma15_4b": Oracle, "gemma3_4b": Blind,
                 "medvlthinker_3b": Blind, "qwen25vl_3b": Blind})
    t = summarize_contrast(paths, sets=SETS, B=50, seed=0)["contrast_tests"].set_index("hypothesis")
    assert list(t.index) == ["H12a", "H12b", "H12c"]
    assert t.loc["H12a", "est"] == 1 and t.loc["H12c", "est"] == 0
    assert (t.p_holm >= t.p).all()
    assert t.loc["H12c", "supported"]                       # identical models: CI inside the +-5 point margin


def test_rerun_is_resumable_and_does_not_duplicate(paths):
    make_contrast(paths)
    _run(paths, {"medgemma4b": Oracle})
    _run(paths, {"medgemma4b": Oracle})
    f = next((paths.outputs / "medgemma4b__opp" / "contrast").glob("results*.jsonl"))
    assert len(f.read_text().splitlines()) == 8


def test_train_overlap_counts_evaluated_images_seen_in_the_train_split():
    ev = pd.DataFrame({"profile": "main", "dataset": "vqa_rad", "split": "test", "image_id": ["a", "a", "b", "c"],
                       "answer_type": ["yesno", "open", "yesno", "yesno"]})
    r = train_overlap(ev, {"vqa_rad": {"a", "c", "z"}}).iloc[0]
    assert (r.images, r.images_in_train_split, r.yesno_questions_on_train_images, r.yesno_questions) == (3, 2, 2, 3)
