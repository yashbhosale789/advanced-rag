from scripts.retrieval_eval import score


def _ranked(*pairs):
    return [{"metadata": {"source": f"D:\\docs\\{src}"}, "text": text} for src, text in pairs]


def test_hit_needs_source_and_span_and_ignores_case_and_punctuation():
    item = {"relevant": [{"source": "a.pdf", "evidence": "Lug-wrench, tighten"}]}
    ranked = _ranked(("b.pdf", "lug wrench tighten"), ("a.pdf", "Use the LUG WRENCH; tighten it"))
    out = score(item, ranked, (1, 3))
    assert out["recall@1"] == 0.0 and out["recall@3"] == 1.0
    assert out["mrr"] == 0.5


def test_multi_span_recall_is_a_fraction_and_success_needs_all():
    item = {"relevant": [{"source": "a.pdf", "evidence": "first"}, {"source": "a.pdf", "evidence": "second"}]}
    out = score(item, _ranked(("a.pdf", "first fact")), (1,))
    assert out["recall@1"] == 0.5 and out["success@1"] == 0.0
    assert score({**item, "any_of": True}, _ranked(("a.pdf", "first")), (1,))["recall@1"] == 1.0


def test_each_stage_gets_its_deeper_k_only_when_the_list_is_that_deep():
    from scripts.retrieval_eval import stage_ks

    assert stage_ks("dense", 20) == (1, 3, 5, 20)
    assert stage_ks("dense", 5) == (1, 3, 5)
    assert stage_ks("reranked", 5) == (1, 3, 5)
