import numpy as np
from pilot_ofdm import cn
from stage3_study import (FREQ, M, L, T, projected_statistics,
                          response_matrix, pool_noise, generate, stage3_profiles)


def test_signed_integer_response_matches_original():
    old = np.exp(-2j*np.pi*np.outer(np.arange(64)/64, np.arange(L)))
    np.testing.assert_allclose(response_matrix(np.arange(L)), old, atol=1e-13)


def test_pooling_reduces_variance_not_model_bias():
    rng = np.random.default_rng(620261113)
    for name in ['sparse_integer', 'fractional_sparse']:
        delays, power = stage3_profiles()[name]
        _, y, _, _, probes = generate(rng, 10000, delays, power, .01)
        _, estimated, _ = projected_statistics(y)
        single, pooled = pool_noise(estimated, 1), pool_noise(estimated, 16)
        assert abs(single.mean()/pooled.mean()-1) < .04
        assert abs(single.var()/pooled.var()/16-1) < .09
        if name == 'sparse_integer':
            assert abs(pooled.mean()/.01-1) < .02
        else:
            assert pooled.mean()/.01 > 2
        assert abs(probes.mean()/.01-1) < .02


def test_integer_noiseless_projection_reconstructs():
    rng = np.random.default_rng(620261114)
    h = cn(rng, (50, L))
    basis = response_matrix(np.arange(L))
    z, vhat, _ = projected_statistics((h @ basis.T)[:, ::4])
    np.testing.assert_allclose(z, h, atol=1e-12)
    assert vhat.max() < 1e-24
