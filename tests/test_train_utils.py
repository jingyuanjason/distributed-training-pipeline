import numpy as np
import pytest
import torch

from implementation.train_utils import get_batch


@pytest.mark.parametrize("vocab_size", [1, 17])
def test_mock_batch(vocab_size):
    data, target = get_batch(
        None, 4, 8, "cpu", vocab_size=vocab_size, mock_data=True,
        rng=np.random.default_rng(42), validate=True,
    )
    for tensor in (data, target):
        assert tensor.shape == (4, 8)
        assert tensor.dtype == torch.long
        assert tensor.device.type == "cpu"
        assert tensor.is_contiguous()
        assert tensor.min() >= 0
        assert tensor.max() < vocab_size
    assert torch.equal(data[:, 1:], target[:, :-1])


def test_mock_batch_rng_reproducibility_and_advancement():
    rng = np.random.default_rng(42)
    state = rng.bit_generator.state
    first = get_batch(None, 4, 8, "cpu", rng=rng, vocab_size=100, mock_data=True)
    second = get_batch(None, 4, 8, "cpu", rng=rng, vocab_size=100, mock_data=True)
    assert not torch.equal(first[0], second[0])
    rng.bit_generator.state = state
    restored = get_batch(None, 4, 8, "cpu", rng=rng, vocab_size=100, mock_data=True)
    for expected, actual in zip(first, restored):
        assert torch.equal(expected, actual)


@pytest.mark.parametrize("vocab_size", [None, 0, -1])
def test_mock_batch_requires_positive_vocab(vocab_size):
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        get_batch(None, 4, 8, "cpu", vocab_size=vocab_size, mock_data=True)


@pytest.mark.parametrize("batch_size,context_length", [(0, 8), (-1, 8), (4, 0), (4, -1)])
def test_mock_batch_requires_positive_dimensions(batch_size, context_length):
    with pytest.raises(ValueError, match="must be positive"):
        get_batch(None, batch_size, context_length, "cpu", vocab_size=17, mock_data=True)


@pytest.mark.parametrize("serial_sampling", [False, True])
def test_dataset_batch_unchanged(serial_sampling):
    tokens = np.arange(100, dtype=np.int64)
    data, target = get_batch(
        tokens, 4, 8, "cpu", serial_sampling=serial_sampling,
        start_idx=2, rng=np.random.default_rng(42), vocab_size=100, validate=True,
    )
    starts = np.arange(2, 34, 8) if serial_sampling else np.random.default_rng(42).integers(2, 92, size=4)
    expected = np.stack([tokens[start:start + 8] for start in starts])
    assert torch.equal(data, torch.from_numpy(expected))
    assert torch.equal(target, data + 1)