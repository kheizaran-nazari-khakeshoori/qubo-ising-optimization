import numpy as np

from vrp_validator import (
    decode_solution,
    validate_vrp_solution,
    is_valid_solution,
    variable_index,
)


def make_valid_x():
    """Construct a feasible VRP assignment for default 4 cities, 2 vehicles.
    Depot=0, A=1,B=2,C=3, capacity 30, demands [0,10,20,15].
    Feasible: vehicle0: A (10) + B (20)=30, vehicle1: C (15)
    """
    cities = ["Depot", "A", "B", "C"]
    n_cities = len(cities)
    n_vehicle = 2
    n_x = n_cities * n_cities * n_vehicle
    x = np.zeros(n_x)
    # vehicle0 pos0 = A (1), pos1 = B (2)
    x[variable_index(1, 0, 0, n_cities, n_vehicle)] = 1
    x[variable_index(2, 1, 0, n_cities, n_vehicle)] = 1
    # vehicle1 pos0 = C (3)
    x[variable_index(3, 0, 1, n_cities, n_vehicle)] = 1
    # Depot fill remaining positions with Depot (0) to satisfy position uniqueness
    # Each position must have one city - we fill empty slots with depot
    x[variable_index(0, 2, 0, n_cities, n_vehicle)] = 1
    x[variable_index(0, 3, 0, n_cities, n_vehicle)] = 1
    x[variable_index(0, 1, 1, n_cities, n_vehicle)] = 1
    x[variable_index(0, 2, 1, n_cities, n_vehicle)] = 1
    x[variable_index(0, 3, 1, n_cities, n_vehicle)] = 1
    return x


def test_valid_solution():
    cities = ["Depot", "A", "B", "C"]
    demand = [0, 10, 20, 15]
    distance = np.array([[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]])
    x = make_valid_x()
    valid, errors, stats = validate_vrp_solution(x, cities, demand, 30, distance)
    assert valid, f"errors: {errors}"
    assert stats["load_vehicle_0"] == 30
    assert stats["load_vehicle_1"] == 15


def test_invalid_customer_visited_twice():
    cities = ["Depot", "A", "B", "C"]
    demand = [0, 10, 20, 15]
    n_cities = 4
    n_vehicle = 2
    n_x = n_cities * n_cities * n_vehicle
    x = np.zeros(n_x)
    # Visit A twice
    x[variable_index(1, 0, 0, n_cities, n_vehicle)] = 1
    x[variable_index(1, 1, 1, n_cities, n_vehicle)] = 1
    x[variable_index(2, 0, 1, n_cities, n_vehicle)] = 1
    valid, errors, _ = validate_vrp_solution(x, cities, demand, 30)
    assert not valid
    assert any("visited 2 times" in e or "visited 0 times" in e for e in errors)


def test_overload():
    cities = ["Depot", "A", "B", "C"]
    demand = [0, 10, 20, 15]
    n_cities = 4
    n_vehicle = 2
    n_x = n_cities * n_cities * n_vehicle
    x = np.zeros(n_x)
    # overload vehicle 0: B(20)+C(15)=35 >30 plus A elsewhere
    x[variable_index(2, 0, 0, n_cities, n_vehicle)] = 1
    x[variable_index(3, 1, 0, n_cities, n_vehicle)] = 1
    x[variable_index(1, 0, 1, n_cities, n_vehicle)] = 1
    # fill depots
    x[variable_index(0, 2, 0, n_cities, n_vehicle)] = 1
    x[variable_index(0, 3, 0, n_cities, n_vehicle)] = 1
    valid, errors, stats = validate_vrp_solution(x, cities, demand, 30)
    assert not valid
    assert any("overload" in e for e in errors)


def test_decode_with_slack():
    """Validator should ignore slack bits appended after 32 vars."""
    cities = ["Depot", "A", "B", "C"]
    n_cities = 4
    n_vehicle = 2
    n_x = n_cities * n_cities * n_vehicle
    x_core = make_valid_x()
    slack = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 0])  # 10 slack bits
    x = np.concatenate([x_core, slack])
    routes = decode_solution(x, cities, n_vehicle)
    assert 1 in routes[0] and 2 in routes[0]


def test_is_valid_wrapper():
    cities = ["Depot", "A", "B", "C"]
    demand = [0, 10, 20, 15]
    x = make_valid_x()
    assert is_valid_solution(x, cities, demand, 30) is True
