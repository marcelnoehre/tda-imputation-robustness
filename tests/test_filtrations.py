import math
import numpy as np
import pytest
from gtda.homology import WeightedRipsPersistence
from sklearn.metrics.pairwise import pairwise_distances

import src.tda as tda
from src.constants import *
from src.config import DIMENSIONS, DTM_PARAMS, KD_PARAMS, MEL_PARAMS


@pytest.fixture(scope='module')
def cloud():
    # noisy circle in R^3 with a few outliers
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 2 * np.pi, 150)
    X = np.c_[np.cos(t), np.sin(t), np.zeros_like(t)] + 0.05 * rng.normal(size=(150, 3))
    return np.vstack([X, rng.uniform(-2, 2, size=(10, 3))])


def test_dtm_matches_giotto(cloud):
    """Our DTM weights in the shared builder reproduce giotto-tda's own 'DTM' weighted Rips filtration."""
    D = pairwise_distances(cloud)
    k = math.ceil(DTM_PARAMS['m'] * len(cloud))
    mel = tda.max_edge_length_for(cloud, DTMC)
    giotto = WeightedRipsPersistence(metric='precomputed', homology_dimensions=DIMENSIONS, max_edge_weight=mel,
                                     collapse_edges=True, weights='DTM', weight_params={'n_neighbors': k, 'p': 2}
                                     ).fit_transform(D[None])
    ours = tda.weighted_rips_complex(cloud, DTMC, mel)
    assert np.allclose(np.sort(ours[0], axis=0), np.sort(giotto[0], axis=0))


def test_dtm_and_kd_share_one_builder(cloud, monkeypatch):
    """DTM and every KD bandwidth run through weighted_rips_complex and differ only in the vertex function."""
    calls = []
    monkeypatch.setattr(tda, 'weighted_rips_complex', lambda data, method, mel, bandwidth=None: calls.append(method))
    methods = [DTMC, *KD_PARAMS['bandwidths']]
    for method in methods:
        tda.TDA[method][FUNCTION](cloud, 1.0, **tda.filtration_params(cloud, method))
    assert calls == methods


@pytest.mark.parametrize('method', [DTMC, KDM, KDM_H025, KDM_H2, KDM_SILVERMAN])
def test_every_vertex_enters_before_the_cutoff(cloud, method):
    D = pairwise_distances(cloud)
    weights = 2 * tda.vertex_weights(cloud, D, method)
    mel = tda.max_edge_length_for(cloud, method)
    assert weights.max() < mel
    # truncation leaves every bar that dies before the cutoff unchanged
    bars = [dg[(dg[:, 1] > dg[:, 0]) & (dg[:, 1] < mel)]
            for dg in (tda.weighted_rips_complex(cloud, method, mel)[0], tda.weighted_rips_complex(cloud, method, np.inf)[0])]
    assert np.allclose(*(b[np.lexsort(b.T[::-1])] for b in bars))


def test_vr_cutoff_is_the_mst_rule(cloud):
    from scipy.sparse.csgraph import minimum_spanning_tree
    D = pairwise_distances(cloud)
    assert tda.max_edge_length_for(cloud, VR) == MEL_PARAMS['mst_scale_factor'] * minimum_spanning_tree(D).max()


def test_kernel_distance_limits(cloud):
    D = pairwise_distances(cloud)
    # h -> infinity: h * d_K tends to the distance to the centroid
    far = tda.kernel_distance_to_measure(D, 1e3)
    assert np.allclose(far, np.linalg.norm(cloud - cloud.mean(axis=0), axis=1), rtol=1e-3)
    # h -> 0: h * d_K tends to the constant h * sqrt(1 - 1/n)
    h = 1e-4
    near = tda.kernel_distance_to_measure(D, h)
    assert np.allclose(near, h * np.sqrt(1 - 1 / len(cloud)), rtol=1e-6)


def test_kernel_distance_is_one_lipschitz(cloud):
    D = pairwise_distances(cloud)
    f = tda.kernel_distance_to_measure(D, tda.kernel_bandwidth(D, cloud.shape[1], KDM))
    assert (np.abs(f[:, None] - f[None, :]) <= D + 1e-9).all()


def test_landscapes_and_images_share_the_original_grid(cloud):
    from src.metrics import landscape_l2_distance, persistence_image_l2_distance
    mel = tda.max_edge_length_for(cloud, VR)
    X = tda.vietoris_rips_complex(cloud, mel)
    Y = tda.vietoris_rips_complex(cloud[:-10], mel)
    pl_X, pl_Y, grid = tda.persistence_landscape(X, Y)
    for dim in DIMENSIONS:
        assert landscape_l2_distance(pl_X[dim], pl_X[dim], grid[dim]) == 0
    # a circle has no H2 features, so only H0 and H1 differ
    assert landscape_l2_distance(pl_X[1], pl_Y[1], grid[1]) > 0
    pi_X, pi_Y, grid = tda.persistence_image(X, Y)
    assert persistence_image_l2_distance(pi_X[0], pi_Y[0], grid[0]) > 0


def test_kd_bandwidth_is_fixed_by_the_original(cloud):
    """A perturbed cloud is filtered with the original's bandwidth, not its own."""
    h = tda.filtration_params(cloud, KDM)['bandwidth']
    moved = cloud * 1.5
    D = pairwise_distances(moved)
    assert np.allclose(tda.vertex_weights(moved, D, KDM, h), tda.kernel_distance_to_measure(D, h))
    assert not np.allclose(tda.vertex_weights(moved, D, KDM), tda.kernel_distance_to_measure(D, h))
