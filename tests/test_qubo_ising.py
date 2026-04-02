"""Test verifying QUBO ↔ Ising ground-state equivalence after fix."""
import importlib.util
import itertools
import pathlib

import numpy as np
import pytest


def load_converter():
    path = pathlib.Path(__file__).parent.parent / "vrp-ising-converter.py"
    spec = importlib.util.spec_from_file_location("vrp_converter", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


@pytest.fixture(scope="module")
def conv():
    return load_converter()


def brute_qubo_min(Q):
    n = Q.shape[0]
    best_E = float("inf")
    best_x = None
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        E = float(x @ Q @ x)
        if E < best_E:
            best_E = E
            best_x = x
    return best_x, best_E


def brute_ising_min(J, h, const):
    n = h.shape[0]
    best_E = float("inf")
    best_s = None
    for bits in itertools.product([-1, 1], repeat=n):
        s = np.array(bits, dtype=float)
        E = float(s @ J @ s + h @ s + const)
        if E < best_E:
            best_E = E
            best_s = s
    return best_s, best_E


def test_energy_equivalence_exhaustive_random(conv):
    """For every x in {0,1}^n, QUBO and Ising energies must match via s=2x-1."""
    rng = np.random.default_rng(0)
    for n in [2, 3, 4]:
        for _ in range(5):
            # Random Q — intentionally upper-triangular to mimic VRP builder
            Q = rng.integers(-5, 6, size=(n, n)).astype(float)
            Q = np.triu(Q)  # upper triangular
            J, h, const = conv.qubo_to_ising(Q)
            for bits in itertools.product([0, 1], repeat=n):
                x = np.array(bits, dtype=float)
                s = 2 * x - 1
                Eq = conv.qubo_energy(x, Q)
                Ei = conv.ising_energy(s, J, h, const)
                assert np.isclose(Eq, Ei), f"n={n} x={x} Eq={Eq} Ei={Ei} Q={Q}"


def test_energy_equivalence_symmetric(conv):
    """Also verify for symmetric Q (QAP instances)."""
    rng = np.random.default_rng(1)
    for n in [3, 4]:
        Q = rng.integers(-3, 4, size=(n, n)).astype(float)
        Q = (Q + Q.T) / 2
        J, h, const = conv.qubo_to_ising(Q)
        for bits in itertools.product([0, 1], repeat=n):
            x = np.array(bits, dtype=float)
            s = 2 * x - 1
            assert np.isclose(conv.qubo_energy(x, Q), conv.ising_energy(s, J, h, const))


def test_ground_state_equivalence(conv):
    """Ground state argmin must be preserved under transformation."""
    rng = np.random.default_rng(42)
    for n in [3, 4]:
        Q = rng.integers(-5, 6, size=(n, n)).astype(float)
        Q = np.triu(Q)
        J, h, const = conv.qubo_to_ising(Q)

        best_x, Eq_min = brute_qubo_min(Q)
        best_s, Ei_min = brute_ising_min(J, h, const)

        # Energies of minima must coincide
        assert np.isclose(Eq_min, Ei_min), f"Q={Q} Eq_min={Eq_min} Ei_min={Ei_min}"

        # Mapping s=2x-1 of QUBO optimum must be Ising optimum
        s_from_x = 2 * best_x - 1
        Ei_from_x = conv.ising_energy(s_from_x, J, h, const)
        assert np.isclose(Ei_from_x, Ei_min)

        x_from_s = (best_s + 1) / 2
        Eq_from_s = conv.qubo_energy(x_from_s, Q)
        assert np.isclose(Eq_from_s, Eq_min)


def test_off_diagonal_division_and_constant(conv):
    """Regression for 2x2 analytical case."""
    Q = np.array([[1, 2], [0, 3]], dtype=float)  # upper triangular
    J, h, const = conv.qubo_to_ising(Q)
    # Symmetrized Q_sym = [[1,1],[1,3]]
    Q_sym = (Q + Q.T) / 2
    # Expected: J off-diag = Q_sym/4 = 0.25, h = 0.5*sum rows = [1.0,2.0], const=0.25*sum+0.25*trace=0.25*6+0.25*4=2.5
    assert np.isclose(J[0, 1], 0.25)
    assert np.isclose(J[1, 0], 0.25)
    assert np.allclose(h, [1.0, 2.0])
    assert np.isclose(const, 2.5)
    # Exhaustive check
    for bits in itertools.product([0, 1], repeat=2):
        x = np.array(bits, dtype=float)
        s = 2 * x - 1
        assert np.isclose(conv.qubo_energy(x, Q), conv.ising_energy(s, J, h, const))


def test_vrp_builder_equivalence(conv):
    """Use actual VRP builder with small instance to ensure equivalence holds."""
    # Build a tiny VRP QUBO (defaults) and spot-check 30 random x
    Q = conv.build_qubo(penalty_A=10, penalty_B=10, penalty_C=5, penalty_D=1)
    J, h, const = conv.qubo_to_ising(Q)
    n = Q.shape[0]
    rng = np.random.default_rng(123)
    for _ in range(30):
        x = rng.integers(0, 2, size=n).astype(float)
        s = 2 * x - 1
        assert np.isclose(conv.qubo_energy(x, Q), conv.ising_energy(s, J, h, const))
