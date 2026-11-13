import numpy as np
from pathlib import Path
import pytest
import torch

from blt.hashes import NGRAMS, RollingHashes, hash_history, hashes_at_ends
from blt.patcher import IncrementalPatcher, calibrate_min_len, max_patch_count, patch_batch, patch_batch_torch


@pytest.mark.parametrize("minimum", [1, 3, 7, 16])
def test_patching_matches_incremental_and_is_causal(minimum):
    rng = np.random.default_rng(19)
    values = rng.integers(0, 256, size=(4, 155), dtype=np.uint8)
    values[:, ::5] = ord(" ")
    batched = patch_batch(values, minimum)
    for row in range(len(values)):
        incremental = IncrementalPatcher(minimum)
        ends = []
        for position, value in enumerate(values[row]):
            assert batched.patch_ids[row, position] == incremental.count
            if incremental.step(int(value)):
                ends.append(position)
        assert ends == batched.ends[row, :batched.counts[row]].tolist()
        assert batched.boundaries[row].nonzero()[0].tolist() == ends
        assert 63 in ends and 127 in ends
    for position in (0, 1, 31, 63, 64, 65, 127, 154):
        changed = values.copy()
        changed[:, position] = ord("!")
        other = patch_batch(changed, minimum)
        np.testing.assert_array_equal(batched.boundaries[:, :position], other.boundaries[:, :position])
        np.testing.assert_array_equal(batched.patch_ids[:, :position + 1], other.patch_ids[:, :position + 1])


def test_calibration_reports_the_nearest_attained_mean():
    samples = [b"One, two three! Four? Five six seven. " * 9, b"more prose, with spaces. " * 7]
    result = calibrate_min_len(samples, 6.0)
    assert 1 <= result.min_len <= 16
    assert 1 <= result.mean <= 16
    assert sum(k * v for k, v in result.histogram.items()) == sum(map(len, samples))
    means = []
    for minimum in range(1, 17):
        lengths = []
        for sample in samples:
            patcher = IncrementalPatcher(minimum)
            last = -1
            for i, byte in enumerate(sample):
                if patcher.step(byte):
                    lengths.append(i - last)
                    last = i
            if last + 1 < len(sample):
                lengths.append(len(sample) - last - 1)
        means.append(sum(lengths) / len(lengths))
    assert abs(result.mean - 6.0) == pytest.approx(min(abs(mean - 6.0) for mean in means))


def test_hashes_are_deterministic_causal_and_match_incremental():
    raw = torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]], dtype=torch.long)
    ends = torch.arange(raw.shape[1]).reshape(1, -1)
    all_hashes = hashes_at_ends(raw, ends, rows=8192, seed=83)
    rolling = RollingHashes(seed=83)
    for i, byte in enumerate(raw[0].tolist()):
        assert all_hashes[0, i].tolist() == rolling.step(byte)
        assert all_hashes[0, i].tolist() == [hash_history(raw[0, :i + 1].tolist(), n, seed=83) for n in NGRAMS]
    changed = raw.clone()
    changed[0, 7:] = 0
    assert torch.equal(all_hashes[:, :7], hashes_at_ends(changed, ends, seed=83)[:, :7])
    assert not torch.equal(all_hashes, hashes_at_ends(raw, ends, seed=84))
    assert hash_history([5], 8) == hash_history([5], 8)


@pytest.mark.parametrize("length", [1, 63, 64, 65, 137, 8192])
def test_block_scans_and_torch_metadata_match(length):
    rng = np.random.default_rng(length)
    data = rng.integers(0, 256, (3, length), dtype=np.uint8)
    data[:, ::4] = ord(" ")
    sizes = np.asarray([length, max(0, length - 3), length // 2])
    capacity = max_patch_count(length, 3)
    reference = patch_batch(data, 3, lengths=sizes, max_patches=capacity)
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        actual = patch_batch_torch(torch.from_numpy(data).to(device), 3,
                                   lengths=torch.from_numpy(sizes).to(device), max_patches=capacity)
        np.testing.assert_array_equal(actual.boundaries.cpu().numpy(), reference.boundaries)
        np.testing.assert_array_equal(actual.patch_ids.cpu().numpy(), reference.patch_ids)
        np.testing.assert_array_equal(actual.ends.cpu().numpy(), reference.ends)
        np.testing.assert_array_equal(actual.counts.cpu().numpy(), reference.counts)
    for row, size in enumerate(sizes):
        incremental = IncrementalPatcher(3)
        expected = [i for i, byte in enumerate(data[row, :size]) if incremental.step(int(byte))]
        assert reference.ends[row, :reference.counts[row]].tolist() == expected
        assert not reference.boundaries[row, size:].any()


def test_real_text_hash_bucket_occupancy():
    directory = Path(__file__).resolve().parents[1] / "discussion"
    sources = [path.read_bytes() for path in sorted(directory.iterdir()) if path.suffix in {".md", ".txt"}]
    ideal = 8192 * (1 - np.exp(-5000 / 8192))
    for n in NGRAMS:
        seen, grams = set(), []
        for source in sources:
            for index in range(len(source) - n + 1):
                gram = source[index:index + n]
                if gram not in seen:
                    seen.add(gram)
                    grams.append(gram)
                    if len(grams) == 5000:
                        break
            if len(grams) == 5000:
                break
        assert len(grams) == 5000
        occupied = len({hash_history(gram, n, seed=0) for gram in grams})
        assert abs(occupied - ideal) / ideal < 0.03


def test_batched_hashes_match_incremental_on_random_bytes():
    rng = np.random.default_rng(220)
    data = rng.integers(0, 256, (4, 191), dtype=np.uint8)
    ends = torch.arange(data.shape[1]).expand(data.shape[0], -1)
    for seed in (0, 71):
        actual = hashes_at_ends(torch.from_numpy(data).long(), ends, seed=seed).numpy()
        for row in range(len(data)):
            rolling = RollingHashes(seed=seed)
            expected = [rolling.step(int(byte)) for byte in data[row]]
            np.testing.assert_array_equal(actual[row], expected)
