"""C engine vs the NumPy reference, plus edge cases (empty book, NaN, strides, threads)."""
import numpy as np
import pytest

from lobimb import reference as ref
from lobimb import engine
from lobimb.data import ASK_OFFSET


def test_version_and_threads():
    assert engine.version() == "1.0.0"
    assert engine.max_threads() >= 1


def test_obi_matches_reference(random_levels):
    np.testing.assert_allclose(engine.obi(random_levels), ref.obi(random_levels), rtol=1e-12)


@pytest.mark.parametrize("depth", [1, 3, 10])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 2.0])
def test_wobi_matches_reference(random_levels, depth, alpha):
    imb, micro = engine.wobi(random_levels, depth=depth, alpha=alpha)
    ref_imb, ref_micro = ref.wobi(random_levels, depth, alpha)
    # atol for imbalances near 0
    np.testing.assert_allclose(imb, ref_imb, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(micro, ref_micro, rtol=1e-12)


def test_fractional_volumes_regression():
    """The old int volumes turned 0.3 vs 0.7 BTC into OBI 0 instead of -0.4."""
    row = np.zeros((1, 40))
    row[0, 1], row[0, ASK_OFFSET + 1] = 0.3, 0.7
    assert engine.obi(row)[0] == pytest.approx(-0.4)
    assert ref.legacy_obi_int(row)[0] == 0.0


def test_obi_bounds_and_empty_book(random_levels):
    out = engine.obi(random_levels)
    assert np.all((out >= -1) & (out <= 1))
    empty = np.zeros((3, 40))
    assert np.all(engine.obi(empty) == 0.0)
    imb, micro = engine.wobi(empty)
    assert np.all(imb == 0.0)


def test_empty_book_microprice_falls_back_to_mid():
    row = np.zeros((1, 40))
    row[0, 0], row[0, ASK_OFFSET] = 100.0, 100.2
    _, micro = engine.wobi(row)
    assert micro[0] == pytest.approx(100.1)


def test_nan_propagates():
    row = np.zeros((1, 40))
    row[0, 1] = np.nan
    assert np.isnan(engine.obi(row)[0])
    assert np.isnan(engine.wobi(row)[0][0])


def test_nan_rows_match_reference(random_levels):
    lv = random_levels[:100].copy()
    lv[0, 1] = np.nan                       # best bid volume
    lv[1, ASK_OFFSET + 2 * 5 + 1] = np.nan  # ask volume, level 5
    lv[2, 4] = np.nan                       # bid price, level 2
    lv[3, :] = np.nan
    for depth in (1, 3, 10):
        imb, micro = engine.wobi(lv, depth=depth, alpha=0.5)
        ref_imb, ref_micro = ref.wobi(lv, depth, 0.5)
        np.testing.assert_allclose(imb, ref_imb, rtol=1e-12, atol=1e-15)  # NaNs compare equal
        np.testing.assert_allclose(micro, ref_micro, rtol=1e-12)
    imb, micro = engine.wobi(lv)
    assert np.isnan(imb[0]) and micro[0] == pytest.approx(0.5 * (lv[0, 0] + lv[0, ASK_OFFSET]))
    assert np.isnan(imb[1]) and np.isnan(imb[3])
    assert not np.isnan(engine.wobi(lv, depth=3)[0][1])  # level 5 not read at depth 3
    np.testing.assert_array_equal(engine.obi(lv), ref.obi(lv))


def test_micro_price_inside_top_of_book(random_levels):
    _, micro = engine.wobi(random_levels, depth=1, alpha=0.0)
    assert np.all(micro >= random_levels[:, 0])
    assert np.all(micro <= random_levels[:, ASK_OFFSET])


def test_large_alpha_converges_to_level1(random_levels):
    imb, _ = engine.wobi(random_levels, depth=10, alpha=50.0)
    np.testing.assert_allclose(imb, engine.obi(random_levels), atol=1e-12)


@pytest.mark.parametrize("kwargs", [{"depth": 0}, {"depth": 11}, {"alpha": -1.0},
                                    {"alpha": float("nan")}])
def test_invalid_arguments(random_levels, kwargs):
    with pytest.raises(engine.EngineError):
        engine.wobi(random_levels[:10], **kwargs)


def test_wrong_shape_rejected():
    with pytest.raises(ValueError):
        engine.obi(np.zeros((5, 39)))


def test_non_contiguous_and_strided_views(random_levels):
    wide = np.zeros((1000, 48))
    wide[:, :40] = random_levels[:1000]
    view = wide[:, :40]  # rows contiguous, stride 48
    np.testing.assert_allclose(engine.obi(view), ref.obi(random_levels[:1000]))
    fortran = np.asfortranarray(random_levels[:1000])
    np.testing.assert_allclose(engine.obi(fortran), ref.obi(random_levels[:1000]))


def test_zero_and_negative_row_strides(random_levels, tmp_path):
    """Zero or negative row strides are copied, not rejected by C."""
    lv = random_levels[:200]
    path = tmp_path / "levels.npy"
    np.save(path, lv)
    mm = np.load(path, mmap_mode="r")
    one = mm[0][None, :]
    assert one.strides[0] == 0
    np.testing.assert_array_equal(engine.obi(one), ref.obi(lv[:1]))
    np.testing.assert_array_equal(engine.wobi(one)[1], engine.wobi(lv[:1])[1])

    rev = lv[::-1]
    assert rev.strides[0] < 0
    np.testing.assert_array_equal(engine.obi(rev), engine.obi(lv)[::-1])
    np.testing.assert_array_equal(engine.wobi(rev)[0], engine.wobi(lv)[0][::-1])

    bc = np.broadcast_to(lv[7], (5, 40))
    assert bc.strides[0] == 0
    np.testing.assert_array_equal(engine.obi(bc), np.full(5, engine.obi(lv[7:8])[0]))
    assert engine.StreamingEngine(bc).obi(4) == engine.obi(lv[7:8])[0]


def test_thread_count_does_not_change_results(random_levels):
    big = np.tile(random_levels, (2, 1))  # above the parallel threshold
    with engine.num_threads(1):
        single = engine.wobi(big)
    multi = engine.wobi(big)
    np.testing.assert_array_equal(single[0], multi[0])
    np.testing.assert_array_equal(single[1], multi[1])


def test_streaming_matches_batch(random_levels):
    stream = engine.StreamingEngine(random_levels, depth=10, alpha=0.5)
    imb, micro = engine.wobi(random_levels, depth=10, alpha=0.5)
    obi = engine.obi(random_levels)
    for i in (0, 1, 777, len(random_levels) - 1):
        assert stream.obi(i) == obi[i]
        assert stream.wobi(i) == (imb[i], micro[i])


def test_streaming_index_out_of_range(random_levels):
    stream = engine.StreamingEngine(random_levels[:10])
    for i in (-1, 10, 10**9):
        with pytest.raises(IndexError):
            stream.obi(i)
        with pytest.raises(IndexError):
            stream.wobi(i)
    assert np.isfinite(stream.obi(9))
