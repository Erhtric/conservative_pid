"""
Tests cover:
- AtomicCounterfactual: string representations, hashing, immutability
- CausalQuery: variable extraction, string representations, hashing, arithmetic operations
- CausalExpression: linear combinations, arithmetic operations, formatting
"""

import pytest
from collections import defaultdict
from cpid.io import AtomicCounterfactual, CausalQuery, CausalExpression


class TestAtomicCounterfactual:
    """Test suite for AtomicCounterfactual class."""

    def test_creation_with_default_interventions(self) -> None:
        """Create an atomic counterfactual with no interventions."""
        cf = AtomicCounterfactual(target_var="Y", target_val=1)
        assert cf.target_var == "Y"
        assert cf.target_val == 1
        assert cf.interventions == {}

    def test_creation_with_interventions(self) -> None:
        """Create an atomic counterfactual with interventions."""
        cf = AtomicCounterfactual(
            target_var="Y", target_val=1, interventions={"X": 1, "Z": 0}
        )
        assert cf.target_var == "Y"
        assert cf.target_val == 1
        assert cf.interventions == {"X": 1, "Z": 0}

    def test_str_no_interventions(self) -> None:
        """String representation without interventions."""
        cf = AtomicCounterfactual(target_var="Y", target_val=1)
        assert str(cf) == "Y=1"

    def test_str_with_single_intervention(self) -> None:
        """String representation with a single intervention."""
        cf = AtomicCounterfactual(target_var="Y", target_val=1, interventions={"X": 1})
        assert str(cf) == "Y_{X=1}=1"

    def test_str_with_multiple_interventions(self) -> None:
        """String representation with multiple interventions (sorted)."""
        cf = AtomicCounterfactual(
            target_var="Y", target_val=0, interventions={"Z": 1, "X": 0, "W": 1}
        )
        result = str(cf)
        assert "Y_" in result
        assert "X=0" in result
        assert "W=1" in result
        assert "Z=1" in result
        assert result.endswith("=0")

    def test_repr_delegates_to_str(self) -> None:
        """Repr should delegate to str."""
        cf = AtomicCounterfactual(target_var="Y", target_val=1, interventions={"X": 1})
        assert repr(cf) == str(cf)

    def test_hash_no_interventions(self) -> None:
        """Hash should be consistent for objects without interventions."""
        cf1 = AtomicCounterfactual(target_var="Y", target_val=1)
        cf2 = AtomicCounterfactual(target_var="Y", target_val=1)
        assert hash(cf1) == hash(cf2)

    def test_hash_with_interventions(self) -> None:
        """Hash should be consistent for identical interventions."""
        cf1 = AtomicCounterfactual(
            target_var="Y", target_val=1, interventions={"X": 1, "Z": 0}
        )
        cf2 = AtomicCounterfactual(
            target_var="Y", target_val=1, interventions={"X": 1, "Z": 0}
        )
        assert hash(cf1) == hash(cf2)

    def test_hash_order_independence(self) -> None:
        """Hash should be the same regardless of intervention dict order."""
        cf1 = AtomicCounterfactual(
            target_var="Y", target_val=1, interventions={"X": 1, "Z": 0}
        )
        cf2 = AtomicCounterfactual(
            target_var="Y", target_val=1, interventions={"Z": 0, "X": 1}
        )
        # Although Python dicts maintain insertion order, our hash uses sorted items
        assert hash(cf1) == hash(cf2)

    def test_hash_different_values(self) -> None:
        """Hash should differ for different target values."""
        cf1 = AtomicCounterfactual(target_var="Y", target_val=0)
        cf2 = AtomicCounterfactual(target_var="Y", target_val=1)
        assert hash(cf1) != hash(cf2)

    def test_hash_different_interventions(self) -> None:
        """Hash should differ for different interventions."""
        cf1 = AtomicCounterfactual(target_var="Y", target_val=1, interventions={"X": 0})
        cf2 = AtomicCounterfactual(target_var="Y", target_val=1, interventions={"X": 1})
        assert hash(cf1) != hash(cf2)

    def test_frozen_dataclass_immutable(self) -> None:
        """Frozen dataclass should prevent mutation."""
        cf = AtomicCounterfactual(target_var="Y", target_val=1)
        with pytest.raises(AttributeError):
            cf.target_val = 0  # type: ignore

    def test_hashable_in_dict(self) -> None:
        """Atomic counterfactuals should be usable as dict keys."""
        cf1 = AtomicCounterfactual(target_var="Y", target_val=1)
        cf2 = AtomicCounterfactual(target_var="Y", target_val=1)
        d = {cf1: "value"}
        assert d[cf2] == "value"


class TestCausalQuery:
    """Test suite for CausalQuery class."""

    def test_creation_minimal(self) -> None:
        """Create a minimal causal query."""
        cf = AtomicCounterfactual(target_var="Y", target_val=1)
        query = CausalQuery(counterfactuals=[cf])
        assert len(query.counterfactuals) == 1
        assert query.evidence == {}

    def test_creation_with_evidence(self) -> None:
        """Create a causal query with evidence."""
        cf = AtomicCounterfactual(target_var="Y", target_val=1)
        query = CausalQuery(counterfactuals=[cf], evidence={"X": 0})
        assert query.evidence == {"X": 0}

    def test_all_variables_single_counterfactual_no_intervention(self) -> None:
        """Extract variables from a single counterfactual without interventions."""
        cf = AtomicCounterfactual(target_var="Y", target_val=1)
        query = CausalQuery(counterfactuals=[cf])
        assert query.all_variables == ["Y"]

    def test_all_variables_single_counterfactual_with_intervention(self) -> None:
        """Extract variables from counterfactual with intervention."""
        cf = AtomicCounterfactual(target_var="Y", target_val=1, interventions={"X": 1})
        query = CausalQuery(counterfactuals=[cf])
        assert set(query.all_variables) == {"Y", "X"}

    def test_all_variables_multiple_counterfactuals(self) -> None:
        """Extract variables from multiple counterfactuals."""
        cf1 = AtomicCounterfactual(target_var="Y", target_val=1, interventions={"X": 1})
        cf2 = AtomicCounterfactual(target_var="Z", target_val=0, interventions={"W": 0})
        query = CausalQuery(counterfactuals=[cf1, cf2])
        assert set(query.all_variables) == {"Y", "X", "Z", "W"}

    def test_all_variables_with_evidence(self) -> None:
        """Extract variables including evidence."""
        cf = AtomicCounterfactual(target_var="Y", target_val=1)
        query = CausalQuery(counterfactuals=[cf], evidence={"X": 0, "Z": 1})
        assert set(query.all_variables) == {"Y", "X", "Z"}

    def test_all_variables_comprehensive(self) -> None:
        """Test comprehensive variable extraction."""
        cf1 = AtomicCounterfactual(
            target_var="Y", target_val=1, interventions={"X": 1, "M": 0}
        )
        cf2 = AtomicCounterfactual(target_var="Z", target_val=0, interventions={"W": 1})
        query = CausalQuery(counterfactuals=[cf1, cf2], evidence={"X": 0})
        # X appears in both interventions and evidence, should not duplicate
        assert set(query.all_variables) == {"Y", "X", "M", "Z", "W"}

    def test_all_variables_sorted(self) -> None:
        """Variables should be sorted."""
        cf = AtomicCounterfactual(
            target_var="Z", target_val=1, interventions={"A": 1, "X": 0}
        )
        query = CausalQuery(counterfactuals=[cf])
        assert query.all_variables == ["A", "X", "Z"]

    def test_str_no_evidence(self) -> None:
        """String representation without evidence."""
        cf = AtomicCounterfactual(target_var="Y", target_val=1)
        query = CausalQuery(counterfactuals=[cf])
        assert str(query) == "P(Y=1)"

    def test_str_with_evidence(self) -> None:
        """String representation with evidence."""
        cf = AtomicCounterfactual(target_var="Y", target_val=1)
        query = CausalQuery(counterfactuals=[cf], evidence={"X": 0})
        assert "P(" in str(query)
        assert "Y=1" in str(query)
        assert "|" in str(query)
        assert "X=0" in str(query)

    def test_str_multiple_counterfactuals(self) -> None:
        """String representation with multiple counterfactuals."""
        cf1 = AtomicCounterfactual(target_var="Y", target_val=1)
        cf2 = AtomicCounterfactual(target_var="Z", target_val=0)
        query = CausalQuery(counterfactuals=[cf1, cf2])
        result = str(query)
        assert "Y=1" in result
        assert "Z=0" in result

    def test_repr_delegates_to_str(self) -> None:
        """Repr should delegate to str."""
        cf = AtomicCounterfactual(target_var="Y", target_val=1)
        query = CausalQuery(counterfactuals=[cf])
        assert repr(query) == str(query)

    def test_hash_identical_queries(self) -> None:
        """Identical queries should have the same hash."""
        cf1 = AtomicCounterfactual(target_var="Y", target_val=1)
        cf2 = AtomicCounterfactual(target_var="Y", target_val=1)
        q1 = CausalQuery(counterfactuals=[cf1])
        q2 = CausalQuery(counterfactuals=[cf2])
        assert hash(q1) == hash(q2)

    def test_hash_with_evidence(self) -> None:
        """Hashes should match for identical queries with evidence."""
        cf = AtomicCounterfactual(target_var="Y", target_val=1)
        q1 = CausalQuery(counterfactuals=[cf], evidence={"X": 0})
        q2 = CausalQuery(counterfactuals=[cf], evidence={"X": 0})
        assert hash(q1) == hash(q2)

    def test_hash_different_counterfactuals(self) -> None:
        """Different counterfactuals should have different hashes."""
        cf1 = AtomicCounterfactual(target_var="Y", target_val=1)
        cf2 = AtomicCounterfactual(target_var="Y", target_val=0)
        q1 = CausalQuery(counterfactuals=[cf1])
        q2 = CausalQuery(counterfactuals=[cf2])
        assert hash(q1) != hash(q2)

    def test_hash_different_evidence(self) -> None:
        """Different evidence should result in different hashes."""
        cf = AtomicCounterfactual(target_var="Y", target_val=1)
        q1 = CausalQuery(counterfactuals=[cf], evidence={"X": 0})
        q2 = CausalQuery(counterfactuals=[cf], evidence={"X": 1})
        assert hash(q1) != hash(q2)

    def test_hashable_in_dict(self) -> None:
        """Queries should be usable as dict keys."""
        cf = AtomicCounterfactual(target_var="Y", target_val=1)
        q1 = CausalQuery(counterfactuals=[cf])
        q2 = CausalQuery(counterfactuals=[cf])
        d = {q1: "value"}
        assert d[q2] == "value"

    def test_frozen_dataclass_immutable(self) -> None:
        """Frozen dataclass should prevent mutation."""
        cf = AtomicCounterfactual(target_var="Y", target_val=1)
        query = CausalQuery(counterfactuals=[cf])
        with pytest.raises(AttributeError):
            query.evidence = {"X": 0}  # type: ignore

    def test_add_two_queries(self) -> None:
        """Adding two queries returns a CausalExpression."""
        q1 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        q2 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=0)]
        )
        expr = q1 + q2
        assert isinstance(expr, CausalExpression)
        assert expr.terms[q1] == 1.0
        assert expr.terms[q2] == 1.0

    def test_add_query_to_expression(self) -> None:
        """Adding a query to an expression."""
        q1 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        q2 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=0)]
        )
        q3 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Z", target_val=1)]
        )
        expr = q1 + q2
        result = q3 + expr
        assert isinstance(result, CausalExpression)
        assert result.terms[q1] == 1.0
        assert result.terms[q2] == 1.0
        assert result.terms[q3] == 1.0

    def test_add_invalid_type(self) -> None:
        """Adding invalid type should return NotImplemented."""
        q = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        result = q.__add__("invalid")  # type: ignore
        assert result is NotImplemented

    def test_sub_two_queries(self) -> None:
        """Subtracting two queries returns a CausalExpression."""
        q1 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        q2 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=0)]
        )
        expr = q1 - q2
        assert isinstance(expr, CausalExpression)
        assert expr.terms[q1] == 1.0
        assert expr.terms[q2] == -1.0

    def test_sub_expression_from_query(self) -> None:
        """Subtracting a CausalExpression from a CausalQuery."""
        q1 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        q2 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=0)]
        )
        q3 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Z", target_val=1)]
        )
        expr = q2 + q3
        result = q1 - expr
        assert isinstance(result, CausalExpression)
        assert result.terms[q1] == 1.0
        assert result.terms[q2] == -1.0
        assert result.terms[q3] == -1.0

    def test_sub_query_from_expression(self) -> None:
        """Subtracting a query from an expression."""
        q1 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        q2 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=0)]
        )
        q3 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Z", target_val=1)]
        )
        expr = q1 + q2
        result = expr - q3
        assert isinstance(result, CausalExpression)
        assert result.terms[q1] == 1.0
        assert result.terms[q2] == 1.0
        assert result.terms[q3] == -1.0

    def test_sub_invalid_type(self) -> None:
        """Subtracting invalid type should return NotImplemented."""
        q = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        result = q.__sub__("invalid")  # type: ignore
        assert result is NotImplemented

    def test_mul_scalar_int(self) -> None:
        """Multiplying query by integer scalar."""
        q = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        expr = q * 2
        assert isinstance(expr, CausalExpression)
        assert expr.terms[q] == 2.0

    def test_mul_scalar_float(self) -> None:
        """Multiplying query by float scalar."""
        q = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        expr = q * 0.5
        assert isinstance(expr, CausalExpression)
        assert expr.terms[q] == 0.5

    def test_mul_invalid_type(self) -> None:
        """Multiplying by invalid type should return NotImplemented."""
        q = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        result = q.__mul__("invalid")  # type: ignore
        assert result is NotImplemented

    def test_rmul_scalar_int(self) -> None:
        """Right multiplication by scalar (e.g., 2 * q)."""
        q = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        expr = 3 * q
        assert isinstance(expr, CausalExpression)
        assert expr.terms[q] == 3.0

    def test_rmul_scalar_float(self) -> None:
        """Right multiplication by float scalar."""
        q = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        expr = 0.25 * q
        assert isinstance(expr, CausalExpression)
        assert expr.terms[q] == 0.25

    def test_neg_query(self) -> None:
        """Negating a query."""
        q = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        expr = -q
        assert isinstance(expr, CausalExpression)
        assert expr.terms[q] == -1.0


# ============================================================================
# CausalExpression Tests
# ============================================================================


class TestCausalExpression:
    """Test suite for CausalExpression class."""

    def test_creation_empty(self) -> None:
        """Create an empty expression."""
        expr = CausalExpression()
        assert expr.terms == {}

    def test_creation_with_terms(self) -> None:
        """Create expression with initial terms."""
        q1 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        q2 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=0)]
        )
        expr = CausalExpression({q1: 1.0, q2: -1.0})
        assert expr.terms[q1] == 1.0
        assert expr.terms[q2] == -1.0

    def test_add_query_to_expression(self) -> None:
        """Adding a query to an expression."""
        q1 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        q2 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=0)]
        )
        expr = CausalExpression({q1: 1.0})
        result = expr + q2
        assert result.terms[q1] == 1.0
        assert result.terms[q2] == 1.0

    def test_add_expression_to_expression(self) -> None:
        """Adding two expressions."""
        q1 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        q2 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=0)]
        )
        expr1 = CausalExpression({q1: 1.0})
        expr2 = CausalExpression({q2: 2.0})
        result = expr1 + expr2
        assert result.terms[q1] == 1.0
        assert result.terms[q2] == 2.0

    def test_add_same_query_combines_weights(self) -> None:
        """Adding same query combines weights."""
        q = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        expr1 = CausalExpression({q: 1.0})
        expr2 = CausalExpression({q: 2.0})
        result = expr1 + expr2
        assert result.terms[q] == 3.0

    def test_add_zero_weights_removed(self) -> None:
        """Zero weights are removed after addition."""
        q1 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        q2 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=0)]
        )
        expr1 = CausalExpression({q1: 1.0})
        expr2 = CausalExpression({q1: -1.0})
        result = expr1 + expr2
        assert q1 not in result.terms

    def test_add_invalid_type(self) -> None:
        """Adding invalid type should return NotImplemented."""
        expr = CausalExpression()
        result = expr.__add__("invalid")  # type: ignore
        assert result is NotImplemented

    def test_sub_query_from_expression(self) -> None:
        """Subtracting a query from an expression."""
        q1 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        q2 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=0)]
        )
        expr = CausalExpression({q1: 1.0})
        result = expr - q2
        assert result.terms[q1] == 1.0
        assert result.terms[q2] == -1.0

    def test_sub_expression_from_expression(self) -> None:
        """Subtracting two expressions."""
        q1 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        q2 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=0)]
        )
        expr1 = CausalExpression({q1: 1.0, q2: 1.0})
        expr2 = CausalExpression({q2: 1.0})
        result = expr1 - expr2
        assert result.terms[q1] == 1.0
        # q2 should be removed because its weight becomes 0 (1.0 - 1.0)
        assert q2 not in result.terms

    def test_sub_invalid_type(self) -> None:
        """Subtracting invalid type should return NotImplemented."""
        expr = CausalExpression()
        result = expr.__sub__("invalid")  # type: ignore
        assert result is NotImplemented

    def test_mul_by_scalar_int(self) -> None:
        """Multiplying expression by integer."""
        q = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        expr = CausalExpression({q: 1.0})
        result = expr * 2
        assert result.terms[q] == 2.0

    def test_mul_by_scalar_float(self) -> None:
        """Multiplying expression by float."""
        q = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        expr = CausalExpression({q: 2.0})
        result = expr * 0.5
        assert result.terms[q] == 1.0

    def test_mul_invalid_type(self) -> None:
        """Multiplying by invalid type should return NotImplemented."""
        expr = CausalExpression()
        result = expr.__mul__("invalid")  # type: ignore
        assert result is NotImplemented

    def test_rmul_scalar_int(self) -> None:
        """Right multiplication by scalar."""
        q = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        expr = CausalExpression({q: 1.0})
        result = 3 * expr
        assert result.terms[q] == 3.0

    def test_rmul_scalar_float(self) -> None:
        """Right multiplication by float scalar."""
        q = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        expr = CausalExpression({q: 1.0})
        result = 2.5 * expr
        assert result.terms[q] == 2.5

    def test_neg_expression(self) -> None:
        """Negating an expression flips all weights."""
        q1 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        q2 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=0)]
        )
        expr = CausalExpression({q1: 1.0, q2: -2.0})
        result = -expr
        assert result.terms[q1] == -1.0
        assert result.terms[q2] == 2.0

    def test_repr_single_term(self) -> None:
        """String representation with single term."""
        q = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        expr = CausalExpression({q: 1.0})
        result = repr(expr)
        assert "+1" in result or "1 * " in result

    def test_repr_multiple_terms(self) -> None:
        """String representation with multiple terms."""
        q1 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        q2 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=0)]
        )
        expr = CausalExpression({q1: 1.0, q2: -1.0})
        result = repr(expr)
        assert "-" in result and "+" in result

    def test_repr_empty_expression(self) -> None:
        """String representation of empty expression."""
        expr = CausalExpression()
        assert repr(expr) == "0.0"

    def test_frozen_dataclass_immutable(self) -> None:
        """Frozen dataclass should prevent mutation."""
        q = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        expr = CausalExpression({q: 1.0})
        with pytest.raises(AttributeError):
            expr.terms = {}  # type: ignore


class TestIntegration:
    """Integration tests across multiple classes."""

    def test_ace_construction(self) -> None:
        """Construct Average Causal Effect: P(Y_{X=1}=1) - P(Y_{X=0}=0)."""
        q1 = CausalQuery(
            counterfactuals=[
                AtomicCounterfactual(
                    target_var="Y", target_val=1, interventions={"X": 1}
                )
            ]
        )
        q2 = CausalQuery(
            counterfactuals=[
                AtomicCounterfactual(
                    target_var="Y", target_val=0, interventions={"X": 0}
                )
            ]
        )
        ace = q1 - q2
        assert isinstance(ace, CausalExpression)
        assert ace.terms[q1] == 1.0
        assert ace.terms[q2] == -1.0

    def test_complex_expression_arithmetic(self) -> None:
        """Test complex arithmetic expressions."""
        q1 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        q2 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=0)]
        )
        q3 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Z", target_val=1)]
        )
        # (q1 + q2) * 0.5 - q3
        expr = (q1 + q2) * 0.5 - q3
        assert expr.terms[q1] == 0.5
        assert expr.terms[q2] == 0.5
        assert expr.terms[q3] == -1.0

    def test_query_in_set(self) -> None:
        """Queries should work in sets."""
        q1 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        q2 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        q3 = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=0)]
        )
        s = {q1, q2, q3}
        assert len(s) == 2  # q1 and q2 should be the same

    def test_query_in_defaultdict(self) -> None:
        """Queries should work as keys in defaultdict."""
        q = CausalQuery(
            counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)]
        )
        d = defaultdict(list)
        d[q].append(1)
        d[q].append(2)
        assert d[q] == [1, 2]

    def test_expression_variable_inspection(self) -> None:
        """CausalExpression should expose target and intervention variable lists."""
        q1 = CausalQuery(
            counterfactuals=[
                AtomicCounterfactual(
                    target_var="Y", target_val=1, interventions={"X": 1}
                )
            ]
        )
        q2 = CausalQuery(
            counterfactuals=[
                AtomicCounterfactual(
                    target_var="Z", target_val=0, interventions={"W": 0}
                )
            ]
        )
        expr = q1 - q2
        assert set(expr.target_variables) == {"Y", "Z"}
        assert set(expr.intervention_variables) == {"X", "W"}

    ac = AtomicCounterfactual("Y", 1)
    q = CausalQuery(counterfactuals=[ac])

    with pytest.raises(AttributeError):
        ac.target_val = 0

    with pytest.raises(AttributeError):
        q.evidence = {"X": 1}
