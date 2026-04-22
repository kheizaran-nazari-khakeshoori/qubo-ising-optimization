"""Bayesian optimization for VRP penalty tuning using scikit-optimize."""
import numpy as np

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


DEFAULT_SPACE = [
    Integer(100, 5000, name="penalty_A"),
    Integer(100, 5000, name="penalty_B"),
    Integer(100, 5000, name="penalty_C"),
    Real(0.1, 10.0, name="penalty_D"),
]


class PenaltyTuner:
    """Wraps gp_minimize for penalty search.

    Example:
        tuner = PenaltyTuner(space=DEFAULT_SPACE, n_calls=30)
        result = tuner.optimize(objective_fn)  # objective_fn(penalties) -> cost
        print(result.x)  # best penalties
    """

    def __init__(self, space=None, n_calls=30, random_state=0, n_initial_points=10):
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize not installed. pip install scikit-optimize")
        self.space = space if space is not None else DEFAULT_SPACE
        self.n_calls = n_calls
        self.random_state = random_state
        self.n_initial_points = n_initial_points
        self.result_ = None

    def optimize(self, objective, verbose=False):
        """Run gp_minimize.
        objective: callable that takes list [A,B,C,D] and returns scalar to minimize.
        """
        @use_named_args(self.space)
        def wrapped(**params):
            x = [params["penalty_A"], params["penalty_B"], params["penalty_C"], params["penalty_D"]]
            return float(objective(x))

        self.result_ = gp_minimize(
            wrapped,
            self.space,
            n_calls=self.n_calls,
            random_state=self.random_state,
            n_initial_points=self.n_initial_points,
            verbose=verbose,
        )
        return self.result_

    def best_penalties(self):
        if self.result_ is None:
            raise RuntimeError("Call optimize first")
        return dict(zip(["penalty_A", "penalty_B", "penalty_C", "penalty_D"], self.result_.x))

    def best_score(self):
        if self.result_ is None:
            raise RuntimeError("Call optimize first")
        return float(self.result_.fun)


def random_search_baseline(objective, n_iter=30, seed=0):
    """Fallback when skopt unavailable: random search."""
    rng = np.random.default_rng(seed)
    best_x = None
    best_val = float("inf")
    for _ in range(n_iter):
        x = [
            int(rng.integers(100, 5000)),
            int(rng.integers(100, 5000)),
            int(rng.integers(100, 5000)),
            float(rng.uniform(0.1, 10.0)),
        ]
        val = float(objective(x))
        if val < best_val:
            best_val = val
            best_x = x
    return best_x, best_val
