import json
from pathlib import Path

import numpy as np
import pytest

from sike.atomics.atomic_state import State
from sike.plasma_utils import boltzmann_dist, saha_dist


@pytest.fixture
def input_states():
    examples_states_filepath = Path(__file__).parent / "data" / "example_states.json"

    with Path.open(examples_states_filepath) as f:
        levels_dict = json.load(f)
        states = [None] * len(levels_dict)
        for i, level_dict in enumerate(levels_dict):
            states[i] = State(**level_dict)
    states[0].ground = True
    states[-1].ground = True
    for i in range(1, len(states) - 1):
        states[i].ground = False
    return states


def test_boltzmann_dist():
    """Test the boltzmann_dist function"""
    # No statistical weight dependence
    Te = [0.1, 100.0, 1000.0]
    num_states = 100
    energies = np.linspace(1e-5, 1000, num_states)
    stat_weights = np.ones(num_states)

    for T in Te:
        dist = boltzmann_dist(T, energies, stat_weights)
        # Check length of output array is correct
        assert len(dist) == num_states

        # Check that densities monotonically decrease
        assert all(dist[:-1] >= dist[1:])

        # Assert that g_normalise does nothing when stat_weights = 1
        dist_gn = boltzmann_dist(T, energies, stat_weights, gnormalise=True)
        assert all(dist_gn == dist)

    # Statistical weight dependence
    Te = [0.1, 100.0, 1000.0]
    num_states = 3
    energies = np.ones(num_states)
    stat_weights = np.array([1, 2, 3])

    for T in Te:
        # Check densities are in reverse order for these statistical weights
        dist = boltzmann_dist(T, energies, stat_weights)
        assert all(dist[:-1] <= dist[1:])

        # Check nornmalised densities are in correct order
        dist_gn = boltzmann_dist(T, energies, stat_weights, gnormalise=True)
        assert all(dist_gn[:-1] >= dist_gn[1:])


def test_saha_dist(input_states):
    """Test saha_dist function

    :param input_states: A minimal set of example atomic states
    """
    Te = [0.0001, 1.0, 5.0, 10.0, 100000.0]
    ne = [1e18, 1e19, 1e20]
    nz_tot = [1e12, 1e16, 1e20]

    # Check a few temperatures and densities
    for T in Te:
        for n in ne:
            for nz in nz_tot:
                dist = saha_dist(T, n, nz, input_states, num_Z=2)
                assert np.isclose(np.sum(dist), nz)

    # Check un-ionized at low Te
    dist = saha_dist(0.00001, 1e20, nz_tot[0], input_states, num_Z=2)
    assert np.isclose(dist[0], nz_tot[0])
    assert np.isclose(dist[1], 0.0)

    # Check fully ionized at high Te
    dist = saha_dist(1e9, 1e20, nz_tot[0], input_states, num_Z=2)
    assert np.isclose(dist[1], nz_tot[0])
    assert np.isclose(dist[0] / nz_tot[0], 0.0)
