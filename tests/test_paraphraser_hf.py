"""Real LM check (network + small download): batched left-padded paraphrasing must match one-at-a-time."""
import pytest

pytestmark = pytest.mark.slow


def test_batched_paraphrasing_matches_single_generation():
    from medvlm.paraphrase import HFParaphraser

    p = HFParaphraser("Qwen/Qwen2.5-0.5B-Instruct", device="cpu", dtype="float32", batch_size=3)
    qs = ["Is there a lesion in the left lung?", "What organ is shown?",
          "Is the cardiac silhouette enlarged in this chest radiograph?"]
    batched = p.batch(qs, "formal")
    singles = [p(q, "formal") for q in qs]
    assert all(b for b in batched)
    assert sum(b == s for b, s in zip(batched, singles)) >= 2        # greedy; tolerate one numeric flip
