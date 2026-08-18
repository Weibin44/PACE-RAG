from pace.config import BATCH_SIZES


def test_fixed_batch_sizes():
    assert BATCH_SIZES.reranker_pair == 8
    assert BATCH_SIZES.llm_generator == 10
    assert BATCH_SIZES.provence_compressor == 4
    assert BATCH_SIZES.splade_encoder == 8

    assert BATCH_SIZES.manifest_dict() == {
            "reranker": 8,
            "llm": 10,
            "provence": 4,
        }