from medvlm.paraphrase import STYLES, TemplateParaphraser, generate, load_valid, validate
from medvlm.schema import read_jsonl

from .conftest import make_items


def test_accepts_a_faithful_paraphrase():
    assert validate("Is there a lesion in the left lung?", "Can you see a lesion within the left lung?",
                    "yesno", "yes") == []


def test_rejects_changed_laterality():
    r = validate("Is there a lesion in the left lung?", "Can you see a lesion within the right lung?", "yesno", "yes")
    assert "protected_terms_changed" in r


def test_rejects_added_negation():
    r = validate("Is the spleen enlarged?", "Is the spleen not enlarged?", "yesno", "yes")
    assert "protected_terms_changed" in r


def test_rejects_dropped_number():
    assert "protected_terms_changed" in validate("How many lesions are in the liver, 2?", "How many lesions are in the liver?", "open", "2")


def test_rejects_near_copy():
    assert "too_similar" in validate("Is the spleen enlarged?", "Is the spleen enlarged ?", "yesno", "no")


def test_rejects_answer_leak_for_open_questions():
    assert "leaks_answer" in validate("Which organ is shown?", "Is the lung the organ that is shown?", "open", "lung")


def test_rejects_yesno_turned_into_open():
    assert "yesno_form_lost" in validate("Is there a lesion in the lung?", "Describe the lesion present in the lung.", "yesno", "yes")


def test_duplicate_paraphrases_do_not_count_twice(paths, cfg):
    items = make_items(paths, n_images=1, per_image=1)
    cfg["paraphrase"]["n"] = 3
    out = paths.manifest_dir("smoke") / "paraphrases.jsonl"
    generate(items, cfg, out, backend=lambda q, s: "Can you see a lesion inside the left lung of this scan?")
    rows = sorted(read_jsonl(out), key=lambda r: r["k"])
    assert [r["valid"] for r in rows] == [True, False, False]
    assert rows[1]["reasons"] == ["duplicate"]


def test_generate_is_resumable_and_records_rejections(paths, cfg):
    items = make_items(paths, n_images=2, per_image=2)
    out = paths.manifest_dir("smoke") / "paraphrases.jsonl"
    cfg["paraphrase"]["n"] = len(STYLES)
    n1 = generate(items, cfg, out, backend=TemplateParaphraser())
    assert n1 == len(items) * len(STYLES)

    calls = []
    generate(items, cfg, out, backend=lambda q, s: calls.append(q) or "x")   # nothing left to generate
    assert calls == []

    rows = list(read_jsonl(out))
    assert any(not r["valid"] for r in rows) and any(r["valid"] for r in rows)
    valid = load_valid(out, n=len(STYLES))
    assert all(p["valid"] for ps in valid.values() for p in ps)
