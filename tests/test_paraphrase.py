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


# Audit round 1 (main, 2026-10-15): real failures the v1 validator let through.
def test_rejects_absence_words_that_invert_a_yes_no_question():
    assert "protected_terms_changed" in validate("Is there small bowel thickening present?",
                                                 "Is small bowel thickening absent?", "yesno", "no")
    assert "protected_terms_changed" in validate("Does the picture contain spleen?", "Does the image lack a spleen?",
                                                 "yesno", "yes")


def test_rejects_added_extent_and_side_words_counted_as_a_multiset():
    assert "protected_terms_changed" in validate("is gastrointestinal present?", "Is gastrointestinal present bilaterally?",
                                                 "yesno", "yes")
    assert "protected_terms_changed" in validate("Which is smaller in this image, liver or right kidney?",
                                                 "Which organ is smaller in this picture, the liver on the right side or the right kidney?",
                                                 "open", "right kidney")


def test_superlative_synonyms_are_equal_but_changes_are_not():
    assert validate("What is the main organ in the image?", "What is the primary organ depicted in the image?",
                    "open", "liver") == []
    assert "protected_terms_changed" in validate("What is the main organ in the image?",
                                                 "What is the biggest organ shown in the picture?", "open", "liver")


def test_rejects_open_question_turned_into_yes_no_and_foreign_characters():
    assert "open_form_lost" in validate("In which two ventricles can calcifications be seen on this CT scan?",
                                        "On this CT scan, can you see any calcifications in the two ventricles?",
                                        "open", "lateral ventricles")
    assert "foreign_characters" in validate("Which sign do you see in the window in this image?",
                                            "Which征象 can be observed within the window in this photograph?",
                                            "open", "mass")
    assert validate("Where is the lesion located?", "Where is the lesion situated’?", "open", "liver") == []


def test_revalidation_applies_new_rules_and_style_exclusion_without_a_model(paths, cfg):
    import json

    from medvlm.paraphrase import generate
    from .conftest import make_items

    items = make_items(paths)[:1]
    it = items[0]
    out = paths.manifest_dir("smoke") / "paraphrases.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg["paraphrase"]["n"] = 2
    cfg["paraphrase"]["exclude_styles"] = ["syntactic"]
    rows = [{"sample_id": it.sample_id, "k": 1, "style": "lexical", "text": it.question.rstrip("?") + " present absent?",
             "valid": True, "reasons": []},
            {"sample_id": it.sample_id, "k": 2, "style": "syntactic", "text": "Could you tell me " + it.question,
             "valid": True, "reasons": []}]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    generate(items, cfg, out, backend=None)               # nothing pending: must not need a backend
    got = {r["k"]: r for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}
    assert not got[1]["valid"] and "protected_terms_changed" in got[1]["reasons"]
    assert not got[2]["valid"] and "style_excluded" in got[2]["reasons"]


def test_yesno_form_check_sees_does_questions():
    # normalize() turns "does" into "doe"; the auxiliary check must use the raw first word
    assert "yesno_form_lost" in validate("Does the bladder appear in this image?", "Which bladder is it?", "yesno", "yes")
    assert validate("Does the bladder appear in this image?", "Is the bladder visible in this image?", "yesno", "yes") == []


def test_added_laterally_and_antonyms_are_rejected():
    assert "protected_terms_changed" in validate("Is the heart enlarged?", "Is the heart laterally enlarged?", "yesno", "no")
    assert "protected_terms_changed" in validate("Does the lung look healthy?", "Does the lung appear unhealthy?", "yesno", "yes")
    assert "protected_terms_changed" in validate("Is this a coronal plane?", "Is this a transverse section?", "yesno", "no")
    assert validate("Is this an axial plane?", "Is this image presented in a transverse plane?", "yesno", "yes") == []
