import numpy as np
import pytest
import torch

from pace.preprocessing.modeling import (
    inference_dtype,
    resolve_device,
    scalar_scores_from_logits,
    splade_pool,
)

def test_cpu_inference_configuration():
    device = resolve_device("cpu")

    assert device == torch.device("cpu")
    assert inference_dtype(device) == torch.float32


def test_auto_device_uses_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    assert resolve_device("auto") == torch.device("cpu")


def test_unavailable_cuda_is_rejected(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    with pytest.raises(
        RuntimeError,
        match="CUDA was requested",
    ):
        resolve_device("cuda:0")

def test_splade_pool_masks_padding():
    logits = torch.tensor(
        [
            [
                [1.0, -1.0],
                [5.0, 5.0],
            ]
        ]
    )
    attention_mask = torch.tensor([[1, 0]])

    pooled = splade_pool(logits, attention_mask)

    expected = torch.tensor(
        [[np.log1p(1.0), 0.0]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(pooled, expected)


@pytest.mark.parametrize(
    ("logits", "expected"),
    [
        (
            torch.tensor([0.2, 0.7]),
            torch.tensor([0.2, 0.7]),
        ),
        (
            torch.tensor([[0.2], [0.7]]),
            torch.tensor([0.2, 0.7]),
        ),
        (
            torch.tensor(
                [
                    [0.1, 0.9],
                    [0.8, 0.2],
                ]
            ),
            torch.tensor([0.9, 0.2]),
        ),
    ],
)
def test_scalar_scores_from_logits(logits, expected):
    actual = scalar_scores_from_logits(logits)
    torch.testing.assert_close(actual, expected)


def test_scalar_scores_reject_invalid_shape():
    with pytest.raises(
        ValueError,
        match="reranker logits",
    ):
        scalar_scores_from_logits(torch.zeros(2, 3, 4))