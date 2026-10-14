import os
import subprocess
import sys

from medvlm.data.sampling import allocate, assign_shuffle_partners, stratified_sample


def _rows(n_ct=60, n_mri=10, n_xr=30):
    rows = []
    for mod, n in (("ct", n_ct), ("mri", n_mri), ("xray", n_xr)):
        for i in range(n):
            rows.append({"sample_id": f"{mod}-{i}", "dataset": "slake", "modality": mod, "region": "",
                         "image_id": f"{mod}-img{i // 3}", "answer_type": "open" if i % 2 else "yesno"})
    return rows


def test_allocate_equal_respects_caps_and_budget():
    q = allocate({("a",): 5, ("b",): 100, ("c",): 100}, 60, "equal")
    assert q[("a",)] == 5                                  # capped at its size, surplus redistributed
    assert q[("b",)] + q[("c",)] == 55 and abs(q[("b",)] - q[("c",)]) <= 1


def test_allocate_never_exceeds_available():
    q = allocate({("a",): 3, ("b",): 4}, 100, "equal")
    assert q == {("a",): 3, ("b",): 4}


def test_stratified_sample_is_balanced_and_deterministic():
    a = stratified_sample(_rows(), 30, ["modality"], seed=7)
    b = stratified_sample(list(reversed(_rows())), 30, ["modality"], seed=7)
    assert [r["sample_id"] for r in a] == [r["sample_id"] for r in b]
    counts = {m: sum(r["modality"] == m for r in a) for m in ("ct", "mri", "xray")}
    assert counts == {"ct": 10, "mri": 10, "xray": 10}


def test_n_none_takes_everything():
    assert len(stratified_sample(_rows(), None, ["modality"], seed=1)) == 100


def test_partners_same_modality_and_different_image():
    rows = _rows()
    assign_shuffle_partners(rows, seed=3)
    mod_of = {r["image_id"]: r["modality"] for r in rows}
    for r in rows:
        assert r["_shuffle_image_id"] != r["image_id"]
        assert mod_of[r["_shuffle_image_id"]] == r["modality"]


def test_partner_choice_is_identical_across_processes():
    """hash() is salted per process; sampling must not depend on it."""
    code = ("from medvlm.data.sampling import assign_shuffle_partners as a;"
            "rows=[{'sample_id':f's{i}','dataset':'d','modality':'m','region':'','image_id':f'i{i}'} for i in range(20)];"
            "a(rows,5);print(','.join(r['_shuffle_image_id'] for r in rows))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": seed}).stdout for seed in ("1", "2", "3")}
    assert len(outs) == 1 and outs.pop().strip()
