"""
Tests for monotonicity constraint filtering in the canonical basis.
Verifies that worlds violating monotonicity constraints are correctly removed.
"""

import numpy as np
import pytest

from canonical import VectorizedCanonicalBasis
from symbolic import CounterfactualTerm, MonotonicityConstraint, Variable


class TestMonotonicityConstraintCreation:
    """Test creation and representation of monotonicity constraints."""

    def test_constraint_ge_creation(self):
        """Test creating a >= constraint."""
        Y = Variable("Y", domain=(0, 1))
        X = Variable("X", domain=(0, 1))

        lhs = Y @ {X: 1}
        rhs = Y @ {X: 0}

        constraint = lhs >= rhs
        assert isinstance(constraint, MonotonicityConstraint)
        assert constraint.operator == "ge"
        assert constraint.lhs == lhs
        assert constraint.rhs == rhs

    def test_constraint_le_creation(self):
        """Test creating a <= constraint."""
        Y = Variable("Y", domain=(0, 1))
        X = Variable("X", domain=(0, 1))

        lhs = Y @ {X: 1}
        rhs = Y @ {X: 0}

        constraint = lhs <= rhs
        assert isinstance(constraint, MonotonicityConstraint)
        assert constraint.operator == "le"

    def test_constraint_repr(self):
        """Test string representation of constraints."""
        Y = Variable("Y", domain=(0, 1))
        X = Variable("X", domain=(0, 1))

        lhs = Y @ {X: 1}
        rhs = Y @ {X: 0}

        constraint_ge = lhs >= rhs
        assert ">=" in repr(constraint_ge)

        constraint_le = lhs <= rhs
        assert "<=" in repr(constraint_le)

    def test_constraint_error_on_different_variables(self):
        """Test that constraints on different variables raise an error."""
        Y = Variable("Y", domain=(0, 1))
        Z = Variable("Z", domain=(0, 1))
        X = Variable("X", domain=(0, 1))

        lhs = Y @ {X: 1}
        rhs = Z @ {X: 0}

        with pytest.raises(ValueError, match="must be on the same variable"):
            _ = lhs >= rhs

    def test_constraint_error_on_non_counterfactual(self):
        """Test that constraints on non-counterfactual terms raise an error."""
        Y = Variable("Y", domain=(0, 1))

        with pytest.raises(TypeError, match="must be between two CounterfactualTerms"):
            _ = Y >= 1


class TestMonotonicityFiltering:
    """Test the filter_worlds method on the canonical basis."""

    def test_filter_simple_monotonicity_binary(self):
        """
        Test filtering with Y_{X=1} >= Y_{X=0} on binary variables.
        This removes all worlds where Y increases when X goes from 1 to 0.
        """
        X = Variable("X", domain=(0, 1))
        Y = Variable("Y", domain=(0, 1))

        basis = VectorizedCanonicalBasis([X, Y])
        n_worlds_before = basis.n_worlds

        # Constraint: Y_{X=1} >= Y_{X=0}
        constraint = (Y @ {X: 1}) >= (Y @ {X: 0})

        basis.filter_worlds([constraint])
        n_worlds_after = basis.n_worlds

        # At least some worlds should be removed
        assert n_worlds_after < n_worlds_before
        assert n_worlds_after > 0

        # Manually verify that all remaining worlds satisfy the constraint
        for world_idx in range(basis.n_worlds):
            term_y_x1 = Y @ {X: 1}
            term_y_x0 = Y @ {X: 0}

            val_y_x1 = basis.evaluate(term_y_x1)[world_idx]
            val_y_x0 = basis.evaluate(term_y_x0)[world_idx]

            assert val_y_x1 >= val_y_x0, (
                f"World {world_idx}: Y_{{X=1}}={val_y_x1} < Y_{{X=0}}={val_y_x0}"
            )

    def test_filter_ternary_monotonicity(self):
        """
        Test filtering with ternary variables.
        Y_{X=2} >= Y_{X=1} >= Y_{X=0} should be enforced.
        """
        X = Variable("X", domain=(0, 1, 2))
        Y = Variable("Y", domain=(0, 1, 2))

        basis = VectorizedCanonicalBasis([X, Y])
        n_worlds_before = basis.n_worlds

        # Constraints: Y is monotonic in X
        constraint1 = (Y @ {X: 1}) >= (Y @ {X: 0})
        constraint2 = (Y @ {X: 2}) >= (Y @ {X: 1})

        basis.filter_worlds([constraint1, constraint2])
        n_worlds_after = basis.n_worlds

        assert n_worlds_after < n_worlds_before
        assert n_worlds_after > 0

        # Verify all remaining worlds satisfy both constraints
        for world_idx in range(basis.n_worlds):
            y_x0 = basis.evaluate(Y @ {X: 0})[world_idx]
            y_x1 = basis.evaluate(Y @ {X: 1})[world_idx]
            y_x2 = basis.evaluate(Y @ {X: 2})[world_idx]

            assert y_x1 >= y_x0
            assert y_x2 >= y_x1

    def test_filter_decreasing_monotonicity(self):
        """
        Test filtering with Y_{X=0} >= Y_{X=1} (decreasing monotonicity).
        """
        X = Variable("X", domain=(0, 1))
        Y = Variable("Y", domain=(0, 1))

        basis = VectorizedCanonicalBasis([X, Y])
        n_worlds_before = basis.n_worlds

        # Constraint: Y_{X=0} >= Y_{X=1} (Y decreases as X increases)
        constraint = (Y @ {X: 0}) >= (Y @ {X: 1})

        basis.filter_worlds([constraint])
        n_worlds_after = basis.n_worlds

        assert n_worlds_after < n_worlds_before
        assert n_worlds_after > 0

        # Verify all remaining worlds satisfy the constraint
        for world_idx in range(basis.n_worlds):
            y_x0 = basis.evaluate(Y @ {X: 0})[world_idx]
            y_x1 = basis.evaluate(Y @ {X: 1})[world_idx]

            assert y_x0 >= y_x1

    def test_filter_le_constraint(self):
        """Test filtering with <= (less than or equal) constraint."""
        X = Variable("X", domain=(0, 1))
        Y = Variable("Y", domain=(0, 1))

        basis = VectorizedCanonicalBasis([X, Y])
        n_worlds_before = basis.n_worlds

        # Constraint: Y_{X=1} <= Y_{X=0}
        constraint = (Y @ {X: 1}) <= (Y @ {X: 0})

        basis.filter_worlds([constraint])
        n_worlds_after = basis.n_worlds

        assert n_worlds_after < n_worlds_before
        assert n_worlds_after > 0

        # Verify all remaining worlds satisfy the constraint
        for world_idx in range(basis.n_worlds):
            y_x1 = basis.evaluate(Y @ {X: 1})[world_idx]
            y_x0 = basis.evaluate(Y @ {X: 0})[world_idx]
            assert y_x1 <= y_x0

    def test_no_filtering_with_empty_constraints(self):
        """Test that empty constraints list does not modify the basis."""
        X = Variable("X", domain=(0, 1))
        Y = Variable("Y", domain=(0, 1))

        basis = VectorizedCanonicalBasis([X, Y])
        n_worlds_before = basis.n_worlds

        basis.filter_worlds([])

        assert basis.n_worlds == n_worlds_before

    def test_multiple_constraints_interaction(self):
        """
        Test that multiple constraints are applied correctly (AND semantics).
        """
        X = Variable("X", domain=(0, 1))
        Y = Variable("Y", domain=(0, 1))
        Z = Variable("Z", domain=(0, 1))

        basis = VectorizedCanonicalBasis([X, Y, Z])
        n_worlds_before = basis.n_worlds

        # Constraints: Y is increasing in X, Z is increasing in Y
        constraint_y = (Y @ {X: 1}) >= (Y @ {X: 0})
        constraint_z = (Z @ {Y: 1}) >= (Z @ {Y: 0})

        basis.filter_worlds([constraint_y, constraint_z])
        n_worlds_after = basis.n_worlds

        assert n_worlds_after < n_worlds_before
        assert n_worlds_after > 0

        # Verify all remaining worlds satisfy both constraints
        for world_idx in range(basis.n_worlds):
            y_x0 = basis.evaluate(Y @ {X: 0})[world_idx]
            y_x1 = basis.evaluate(Y @ {X: 1})[world_idx]
            z_y0 = basis.evaluate(Z @ {Y: 0})[world_idx]
            z_y1 = basis.evaluate(Z @ {Y: 1})[world_idx]

            assert y_x1 >= y_x0
            assert z_y1 >= z_y0

    def test_filter_world_count_reduction(self):
        """
        Test that the number of removed worlds is as expected.
        With monotonicity Y_{X=1} >= Y_{X=0}, we remove worlds where Y depends negatively on X.
        For binary X, Y: 8 total worlds, expect exactly 2 to be removed
        (the two worlds where Y(., 0) < Y(., 1)).
        """
        X = Variable("X", domain=(0, 1))
        Y = Variable("Y", domain=(0, 1))

        basis = VectorizedCanonicalBasis([X, Y])
        # 8 total worlds: 2^3 (2 vars * 2 domains each, but Y has 2 parent configs from X)

        constraint = (Y @ {X: 1}) >= (Y @ {X: 0})
        basis.filter_worlds([constraint])

        # The constraint removes worlds where Y(X=0)=1 and Y(X=1)=0
        # which is 1 function (out of 4 possible for Y), affecting 2 worlds
        # So we expect 8 - 2 = 6 worlds remaining
        assert basis.n_worlds == 6

    def test_filter_preserves_evaluation_consistency(self):
        """
        Test that after filtering, the evaluate() method still works correctly
        and returns correctly shaped arrays.
        """
        X = Variable("X", domain=(0, 1))
        Y = Variable("Y", domain=(0, 1))

        basis = VectorizedCanonicalBasis([X, Y])
        constraint = (Y @ {X: 1}) >= (Y @ {X: 0})
        basis.filter_worlds([constraint])

        # Evaluate various terms and check array shapes
        val_x = basis.evaluate(X @ {})
        val_y = basis.evaluate(Y @ {})
        val_y_x0 = basis.evaluate(Y @ {X: 0})
        val_y_x1 = basis.evaluate(Y @ {X: 1})

        n_worlds = basis.n_worlds

        assert val_x.shape == (n_worlds,)
        assert val_y.shape == (n_worlds,)
        assert val_y_x0.shape == (n_worlds,)
        assert val_y_x1.shape == (n_worlds,)

    def test_filter_with_complex_interventions(self):
        """
        Test filtering with more complex counterfactual terms (nested interventions).
        """
        Z = Variable("Z", domain=(0, 1))
        X = Variable("X", domain=(0, 1))
        Y = Variable("Y", domain=(0, 1))

        basis = VectorizedCanonicalBasis([Z, X, Y])
        n_worlds_before = basis.n_worlds

        # Constraint: Y_{X=1} >= Y_{X=0} (Y monotonic in X)
        constraint = (Y @ {X: 1}) >= (Y @ {X: 0})

        basis.filter_worlds([constraint])
        n_worlds_after = basis.n_worlds

        assert n_worlds_after < n_worlds_before
        assert n_worlds_after > 0

        # Verify all remaining worlds satisfy the constraint
        for world_idx in range(basis.n_worlds):
            y_x0 = basis.evaluate(Y @ {X: 0})[world_idx]
            y_x1 = basis.evaluate(Y @ {X: 1})[world_idx]
            assert y_x1 >= y_x0
