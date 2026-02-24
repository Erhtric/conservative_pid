import numpy as np
import pytest
import pandas as pd
from canonical import VectorizedCanonicalBasis
from solver import LPSolver
from symbolic import Event, P, Variable


def test_solver_expression_basic():
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))
    data = pd.DataFrame({"X": [0, 1], "Y": [0, 1], "probability": [0.5, 0.5]})
    basis = VectorizedCanonicalBasis([X, Y])
    solver = LPSolver(basis, data, [X, Y])
    ev1 = Y == 0
    ev2 = Y == 1
    q1 = P(ev1)
    q2 = P(ev2)
    expr = q1 + q2

    lb, ub = solver.solve(expr)

    assert bool(np.isclose(lb, 1.0))
    assert bool(np.isclose(ub, 1.0))


def test_solver_expression_ate_bounds():
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))
    data = pd.DataFrame({"X": [0, 1], "Y": [0, 1], "probability": [0.5, 0.5]})

    basis = VectorizedCanonicalBasis([X, Y])
    solver = LPSolver(basis, data, [X, Y])

    t1 = Y @ {X: 1}
    t0 = Y @ {X: 0}

    q1 = P((t1 == 1))
    q0 = P((t0 == 1))

    ate = q1 - q0

    lb, ub = solver.solve(ate)

    assert float(lb) >= -1.0
    assert float(ub) <= 1.0


def test_solver_expression_conditional():
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    data = pd.DataFrame({"X": [0, 1], "Y": [0, 1], "probability": [0.5, 0.5]})
    basis = VectorizedCanonicalBasis([X, Y])
    solver = LPSolver(basis, data, [X, Y])

    q1 = P((Y == 1), evidence=(X == 0))
    q2 = P((Y == 0), evidence=(X == 0))

    expr = q1 + q2

    lb, ub = solver.solve(expr)
    assert bool(np.isclose(lb, 1.0))
    assert bool(np.isclose(ub, 1.0))


def test_solver_expression_error_mixed_evidence():
    X = Variable("X", domain=(0, 1))

    q1 = P((X == 0))
    q2 = P((X == 0), evidence=(X == 1))

    expr = q1 + q2

    data = pd.DataFrame({"X": [0, 1], "probability": [0.5, 0.5]})
    basis = VectorizedCanonicalBasis([X])
    solver = LPSolver(basis, data, [X])

    with pytest.raises(ValueError, match="must share the same evidence"):
        solver.solve(expr)

    q3 = P((X == 0), evidence=(X == 0))
    expr2 = q2 + q3
    with pytest.raises(ValueError, match="must share the same evidence"):
        solver.solve(expr2)


class TestObjectiveFunctionConstruction:
    """Test that the objective function is correctly constructed with proper query filtering."""

    def test_objective_mask_simple_event(self):
        """Test that the mask correctly identifies worlds matching a simple event."""
        X = Variable("X", domain=(0, 1))
        Y = Variable("Y", domain=(0, 1))

        basis = VectorizedCanonicalBasis([X, Y])

        # Get mask for Y == 0
        event_y0 = Y == 0
        mask = basis.get_mask(event_y0)

        # Verify the mask is a boolean array
        assert mask.dtype == bool
        assert len(mask) == basis.n_worlds

        # Verify that worlds in the mask actually satisfy the event
        for world_idx in np.where(mask)[0]:
            y_val = basis.evaluate(Y @ {})[world_idx]
            assert y_val == 0, f"World {world_idx} in mask has Y != 0"

    def test_objective_mask_counterfactual_event(self):
        """Test that the mask correctly identifies worlds matching a counterfactual event."""
        X = Variable("X", domain=(0, 1))
        Y = Variable("Y", domain=(0, 1))

        basis = VectorizedCanonicalBasis([X, Y])

        # Get mask for Y_{X=1} == 1
        cf_term = Y @ {X: 1}
        event = cf_term == 1
        mask = basis.get_mask(event)

        assert mask.dtype == bool
        assert len(mask) == basis.n_worlds

        # Verify worlds in the mask satisfy the counterfactual event
        for world_idx in np.where(mask)[0]:
            cf_val = basis.evaluate(cf_term)[world_idx]
            assert cf_val == 1, f"World {world_idx} has Y_{{X=1}} != 1"

    def test_objective_indices_extracted_correctly(self):
        """Test that the indices of satisfying worlds are correctly extracted."""
        X = Variable("X", domain=(0, 1))
        Y = Variable("Y", domain=(0, 1))

        basis = VectorizedCanonicalBasis([X, Y])

        event = Y == 1
        mask = basis.get_mask(event)
        indices = np.where(mask)[0]

        # At least some worlds should satisfy Y == 1
        assert len(indices) > 0
        assert len(indices) < basis.n_worlds

        # All extracted indices should be valid
        assert np.all(indices < basis.n_worlds)
        assert np.all(indices >= 0)

    def test_objective_construction_single_term(self):
        """Test objective function construction for a single query term."""
        X = Variable("X", domain=(0, 1))
        Y = Variable("Y", domain=(0, 1))

        data = pd.DataFrame(
            {
                "X": [0, 0, 1, 1],
                "Y": [0, 1, 0, 1],
                "probability": [0.25, 0.25, 0.25, 0.25],
            }
        )
        basis = VectorizedCanonicalBasis([X, Y])
        solver = LPSolver(basis, data, [X, Y])

        # Create a single query
        query = P(Y == 1)

        # Solve to get the LP problem
        lb, ub, problems = solver.solve(query, return_problems=True)

        # Verify that both problems exist
        assert problems["lower"] is not None
        assert problems["upper"] is not None

        # Verify that the bounds are valid probabilities
        assert 0 <= lb <= 1
        assert 0 <= ub <= 1
        assert lb <= ub

    def test_objective_construction_multiple_terms(self):
        """Test objective function construction for multiple query terms with correct weights."""
        X = Variable("X", domain=(0, 1))
        Y = Variable("Y", domain=(0, 1))

        data = pd.DataFrame(
            {
                "X": [0, 0, 1, 1],
                "Y": [0, 1, 0, 1],
                "probability": [0.25, 0.25, 0.25, 0.25],
            }
        )
        basis = VectorizedCanonicalBasis([X, Y])
        solver = LPSolver(basis, data, [X, Y])

        # Create an expression with positive and negative terms
        q1 = P(Y == 1)
        q2 = P(Y == 0)
        expr = 2.0 * q1 - 1.0 * q2

        lb, ub, problems = solver.solve(expr, return_problems=True)

        # The bounds should reflect the weighted combination
        assert problems["lower"] is not None
        assert problems["upper"] is not None

        # Verify objective value is within expected range
        # 2*P(Y=1) - P(Y=0) when P(Y=1) + P(Y=0) = 1
        # If P(Y=1) = 1, then 2*1 - 0 = 2
        # If P(Y=1) = 0, then 2*0 - 1 = -1
        assert lb >= -1.0
        assert ub <= 2.0

    def test_objective_with_counterfactual_interventions(self):
        """Test objective function with counterfactual interventions."""
        X = Variable("X", domain=(0, 1))
        Y = Variable("Y", domain=(0, 1))

        data = pd.DataFrame({"X": [0, 1], "Y": [0, 1], "probability": [0.5, 0.5]})
        basis = VectorizedCanonicalBasis([X, Y])
        solver = LPSolver(basis, data, [X, Y])

        # Query about counterfactual outcomes
        cf_y_x1 = Y @ {X: 1}
        cf_y_x0 = Y @ {X: 0}

        q1 = P(cf_y_x1 == 1)
        q0 = P(cf_y_x0 == 1)

        expr = q1 - q0  # ATE

        lb, ub, problems = solver.solve(expr, return_problems=True)

        # ATE bounds should be in [-1, 1]
        assert -1.0 <= lb <= 1.0
        assert -1.0 <= ub <= 1.0

    def test_objective_complementary_events_sum_to_one(self):
        """Test that complementary events sum to 1.0."""
        X = Variable("X", domain=(0, 1))
        Y = Variable("Y", domain=(0, 1))

        data = pd.DataFrame({"X": [0, 1], "Y": [0, 1], "probability": [0.5, 0.5]})
        basis = VectorizedCanonicalBasis([X, Y])
        solver = LPSolver(basis, data, [X, Y])

        # Complementary events
        q_y0 = P(Y == 0)
        q_y1 = P(Y == 1)

        expr = q_y0 + q_y1

        lb, ub = solver.solve(expr)

        # Should sum to exactly 1.0
        assert np.isclose(lb, 1.0)
        assert np.isclose(ub, 1.0)

    def test_objective_with_conditional_query(self):
        """Test objective function construction for conditional queries."""
        X = Variable("X", domain=(0, 1))
        Y = Variable("Y", domain=(0, 1))

        data = pd.DataFrame(
            {
                "X": [0, 0, 1, 1],
                "Y": [0, 1, 0, 1],
                "probability": [0.25, 0.25, 0.25, 0.25],
            }
        )
        basis = VectorizedCanonicalBasis([X, Y])
        solver = LPSolver(basis, data, [X, Y])

        # Conditional probability
        query = P(Y == 1, evidence=(X == 1))

        lb, ub, problems = solver.solve(query, return_problems=True)

        # Verify that conditional probability is computed
        assert 0 <= lb <= 1
        assert 0 <= ub <= 1
        assert problems["lower"] is not None
        assert problems["upper"] is not None

    def test_objective_mask_conjunction_events(self):
        """Test that conjunction of events correctly filters worlds."""
        X = Variable("X", domain=(0, 1))
        Y = Variable("Y", domain=(0, 1))

        basis = VectorizedCanonicalBasis([X, Y])

        # Event: X == 1 AND Y == 1
        event = (X == 1) & (Y == 1)
        mask = basis.get_mask(event)

        # Verify all worlds in mask satisfy both conditions
        for world_idx in np.where(mask)[0]:
            x_val = basis.evaluate(X @ {})[world_idx]
            y_val = basis.evaluate(Y @ {})[world_idx]
            assert x_val == 1 and y_val == 1

    def test_objective_problem_return_structure(self):
        """Test that returned problems have correct structure."""
        X = Variable("X", domain=(0, 1))
        Y = Variable("Y", domain=(0, 1))

        data = pd.DataFrame({"X": [0, 1], "Y": [0, 1], "probability": [0.5, 0.5]})
        basis = VectorizedCanonicalBasis([X, Y])
        solver = LPSolver(basis, data, [X, Y])

        query = P(Y == 1)

        # Without return_problems
        result1 = solver.solve(query)
        assert isinstance(result1, tuple)
        assert len(result1) == 2

        # With return_problems
        result2 = solver.solve(query, return_problems=True)
        assert isinstance(result2, tuple)
        assert len(result2) == 3

        lb1, ub1 = result1
        lb2, ub2, problems = result2

        assert np.isclose(lb1, lb2)
        assert np.isclose(ub1, ub2)
        assert isinstance(problems, dict)
        assert "lower" in problems
        assert "upper" in problems
