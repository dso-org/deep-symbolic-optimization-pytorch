"""Tests for skeleton expression functionality."""

import numpy as np
import pytest

from dso.skeleton import (
    SkeletonExpression,
    SkeletonOptimizer,
    parse_skeleton,
)


class TestParseSkeleton:
    """Tests for skeleton expression parsing."""

    def test_parse_linear_fourier(self):
        """Test parsing linear Fourier basis."""
        metadata = self._extracted_from_test_parse_nonlinear_exponential_3(
            "a*cos(x1) + b*sin(x1) + c", True
        )
        assert len(metadata["basis_funcs"]) == 3

    def test_parse_nonlinear_exponential(self):
        """Test parsing nonlinear exponential expression."""
        metadata = self._extracted_from_test_parse_nonlinear_exponential_3(
            "a*exp(b*x1) + c", False
        )
        assert "eval_func" in metadata

    def _extracted_from_test_parse_nonlinear_exponential_3(self, arg0, arg1):
        result = self._extracted_from_test_parse_power_law_nonlinear_3(arg0, 1, arg1)
        assert result["var_names"] == ["x1"]
        assert result["n_coefs"] == 3
        return result

    def test_parse_multiple_variables(self):
        """Test parsing expression with multiple variables."""
        metadata = self._extracted_from_test_parse_power_law_nonlinear_3(
            "a*x1 + b*x2*x3 + c", 3, True
        )
        assert set(metadata["var_names"]) == {"x1", "x2", "x3"}

    def test_parse_polynomial(self):
        """Test parsing polynomial expression."""
        metadata = self._extracted_from_test_parse_power_law_nonlinear_3(
            "a*x1**2 + b*x1 + c", 1, True
        )

    # TODO Rename this here and in `test_parse_linear_fourier`, `test_parse_nonlinear_exponential`, `test_parse_multiple_variables`, `test_parse_polynomial` and `test_parse_power_law_nonlinear`
    def test_parse_power_law_nonlinear(self):
        """Test parsing nonlinear power law."""
        metadata = self._extracted_from_test_parse_power_law_nonlinear_3(
            "a*x1**b + c", 1, False
        )

    # TODO Rename this here and in `test_parse_linear_fourier`, `test_parse_nonlinear_exponential`, `test_parse_multiple_variables`, `test_parse_polynomial` and `test_parse_power_law_nonlinear`
    def _extracted_from_test_parse_power_law_nonlinear_3(self, arg0, n_input_vars, arg2):
        result = parse_skeleton(arg0, n_input_vars=n_input_vars)
        assert result["is_linear"] is arg2
        assert set(result["coef_names"]) == {"a", "b", "c"}
        return result

    def test_parse_invalid_syntax(self):
        """Test error handling for invalid syntax."""
        with pytest.raises(ValueError, match="Failed to parse"):
            parse_skeleton("a*cos(x1) + b*sin(", n_input_vars=1)

    def test_parse_invalid_variable(self):
        """Test error handling for invalid variable reference."""
        with pytest.raises(ValueError, match="invalid variables"):
            parse_skeleton("a*x99 + b", n_input_vars=3)

    def test_parse_no_coefficients(self):
        """Test error handling when no coefficients present."""
        with pytest.raises(ValueError, match="no coefficients"):
            parse_skeleton("x1 + x2", n_input_vars=2)

    def test_parse_no_free_symbols(self):
        """Test error handling when no free symbols."""
        with pytest.raises(ValueError, match="no free symbols"):
            parse_skeleton("2 + 3", n_input_vars=1)


class TestSkeletonExpression:
    """Tests for SkeletonExpression token class."""

    def test_create_linear_skeleton(self):
        """Test creating linear skeleton token."""
        skeleton = SkeletonExpression(
            name="fourier1",
            expression_str="a*cos(x1) + b*sin(x1) + c",
            n_input_vars=1,
        )

        assert skeleton.name == "fourier1"
        assert skeleton.arity == 0
        assert skeleton.coef is None
        assert skeleton.metadata["is_linear"] is True

    def test_create_nonlinear_skeleton(self):
        """Test creating nonlinear skeleton token."""
        skeleton = SkeletonExpression(
            name="exp_decay",
            expression_str="a*exp(-b*x1) + c",
            n_input_vars=1,
        )

        assert skeleton.name == "exp_decay"
        assert skeleton.metadata["is_linear"] is False

    def test_skeleton_repr_without_coef(self):
        """Test string representation before optimization."""
        skeleton = SkeletonExpression(
            name="test",
            expression_str="a*x1 + b",
            n_input_vars=1,
        )

        repr_str = repr(skeleton)
        assert "test" in repr_str
        assert "a*x1 + b" in repr_str

    def test_skeleton_repr_with_coef(self):
        """Test string representation after optimization."""
        skeleton = SkeletonExpression(
            name="test",
            expression_str="a*x1 + b",
            n_input_vars=1,
            coef=np.array([2.5, 1.0]),
        )

        repr_str = repr(skeleton)
        assert "2.5" in repr_str
        assert "1.0" in repr_str

    def test_eval_without_optimization_raises(self):
        """Test that evaluation without coefficients raises error."""
        skeleton = SkeletonExpression(
            name="test",
            expression_str="a*x1 + b",
            n_input_vars=1,
        )

        X = np.array([[1.0], [2.0], [3.0]])

        with pytest.raises(ValueError, match="not been optimized"):
            skeleton(X)


class TestSkeletonOptimizer:
    """Tests for skeleton coefficient optimization."""

    def test_optimize_linear_known_data(self):
        """Test optimization on known linear data."""
        # Generate data: y = 2*cos(x) + 3*sin(x) + 1
        np.random.seed(42)
        X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
        y = 2 * np.cos(X[:, 0]) + 3 * np.sin(X[:, 0]) + 1

        # Create skeleton
        skeleton = SkeletonExpression(
            name="fourier",
            expression_str="a*cos(x1) + b*sin(x1) + c",
            n_input_vars=1,
        )

        # Optimize
        optimizer = SkeletonOptimizer()
        optimized = optimizer.fit(skeleton, X, y)

        # Check coefficients are close to true values
        assert optimized.coef is not None
        assert len(optimized.coef) == 3

        # Find which coefficient corresponds to which term
        # Since coefficients are sorted alphabetically: a, b, c
        # a*cos(x1) -> a should be ~2
        # b*sin(x1) -> b should be ~3
        # c -> c should be ~1
        assert np.abs(optimized.coef[0] - 2.0) < 0.01  # a
        assert np.abs(optimized.coef[1] - 3.0) < 0.01  # b
        assert np.abs(optimized.coef[2] - 1.0) < 0.01  # c

    def test_optimize_linear_polynomial(self):
        """Test optimization on polynomial data."""
        # Generate data: y = 3*x^2 + 2*x + 1
        np.random.seed(42)
        X = np.linspace(-2, 2, 50).reshape(-1, 1)
        y = 3 * X[:, 0] ** 2 + 2 * X[:, 0] + 1

        # Create skeleton
        skeleton = SkeletonExpression(
            name="poly2",
            expression_str="a*x1**2 + b*x1 + c",
            n_input_vars=1,
        )

        # Optimize
        optimizer = SkeletonOptimizer()
        optimized = optimizer.fit(skeleton, X, y)

        # Check coefficients
        assert np.abs(optimized.coef[0] - 3.0) < 0.01  # a
        assert np.abs(optimized.coef[1] - 2.0) < 0.01  # b
        assert np.abs(optimized.coef[2] - 1.0) < 0.01  # c

    def test_optimize_nonlinear_exponential(self):
        """Test optimization on nonlinear exponential data."""
        # Generate data: y = 5*exp(-0.1*x) + 2
        np.random.seed(42)
        X = np.linspace(0, 10, 50).reshape(-1, 1)
        y = 5 * np.exp(-0.1 * X[:, 0]) + 2

        # Create skeleton
        skeleton = SkeletonExpression(
            name="exp_decay",
            expression_str="a*exp(-b*x1) + c",
            n_input_vars=1,
        )

        # Optimize
        optimizer = SkeletonOptimizer(
            nonlinear_optimizer_params={
                "method": "L-BFGS-B",
                "options": {"maxiter": 200},
            }
        )
        optimized = optimizer.fit(skeleton, X, y)

        # Check coefficients are reasonable (nonlinear optimization is harder)
        # Coefficients: a (~5), b (~0.1), c (~2)
        assert optimized.coef is not None
        assert len(optimized.coef) == 3

        # Check that predictions are close
        y_pred = optimized(X)
        mse = np.mean((y - y_pred) ** 2)
        assert mse < 0.1, f"MSE {mse} too high"

    def test_optimize_multiple_variables(self):
        """Test optimization with multiple variables."""
        # Generate data: y = 2*x1 + 3*x2 + 1
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 2 * X[:, 0] + 3 * X[:, 1] + 1

        # Create skeleton
        skeleton = SkeletonExpression(
            name="linear2",
            expression_str="a*x1 + b*x2 + c",
            n_input_vars=2,
        )

        # Optimize
        optimizer = SkeletonOptimizer()
        optimized = optimizer.fit(skeleton, X, y)

        # Check coefficients
        assert np.abs(optimized.coef[0] - 2.0) < 0.1  # a
        assert np.abs(optimized.coef[1] - 3.0) < 0.1  # b
        assert np.abs(optimized.coef[2] - 1.0) < 0.1  # c

    def test_execution_after_optimization(self):
        """Test that skeleton can be executed after optimization."""
        # Generate data
        np.random.seed(42)
        X_train = np.linspace(0, 2 * np.pi, 50).reshape(-1, 1)
        y_train = 2 * np.cos(X_train[:, 0]) + 1

        # Create and optimize skeleton
        skeleton = SkeletonExpression(
            name="test",
            expression_str="a*cos(x1) + b",
            n_input_vars=1,
        )
        optimizer = SkeletonOptimizer()
        optimized = optimizer.fit(skeleton, X_train, y_train)

        # Test on new data
        X_test = np.array([[0.0], [np.pi / 2], [np.pi]])
        y_test = optimized(X_test)

        # Check predictions
        expected = np.array([2 * np.cos(0) + 1, 2 * np.cos(np.pi / 2) + 1, 2 * np.cos(np.pi) + 1])
        assert y_test.shape == (3,)
        np.testing.assert_allclose(y_test, expected, atol=0.1)

    def test_optimization_with_noise(self):
        """Test optimization on noisy data."""
        # Generate noisy data
        np.random.seed(42)
        X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
        y_clean = 2 * np.cos(X[:, 0]) + 3 * np.sin(X[:, 0]) + 1
        y_noisy = y_clean + 0.1 * np.random.randn(100)

        # Create and optimize skeleton
        skeleton = SkeletonExpression(
            name="fourier",
            expression_str="a*cos(x1) + b*sin(x1) + c",
            n_input_vars=1,
        )
        optimizer = SkeletonOptimizer()
        optimized = optimizer.fit(skeleton, X, y_noisy)

        # Coefficients should still be close to true values
        assert np.abs(optimized.coef[0] - 2.0) < 0.2
        assert np.abs(optimized.coef[1] - 3.0) < 0.2
        assert np.abs(optimized.coef[2] - 1.0) < 0.2


class TestSkeletonComplexity:
    """Tests for skeleton complexity computation."""

    def test_complexity_before_optimization(self):
        """Test that complexity is 1 before optimization."""
        skeleton = SkeletonExpression(
            name="test",
            expression_str="a*x1 + b",
            n_input_vars=1,
        )
        assert skeleton.complexity == 1

    def test_complexity_after_optimization(self):
        """Test that complexity equals number of coefficients after optimization."""
        skeleton = SkeletonExpression(
            name="test",
            expression_str="a*cos(x1) + b*sin(x1) + c",
            n_input_vars=1,
            coef=np.array([2.0, 3.0, 1.0]),
        )
        assert skeleton.complexity == 3


class TestSkeletonEdgeCases:
    """Tests for edge cases and error handling."""

    def test_constant_skeleton(self):
        """Test skeleton that is just a constant."""
        # Skeleton with no variables, only coefficient
        skeleton = SkeletonExpression(
            name="const",
            expression_str="a",
            n_input_vars=1,
        )

        # Generate data: y = 5.0
        X = np.random.randn(50, 1)
        y = np.full(50, 5.0)

        # Optimize
        optimizer = SkeletonOptimizer()
        optimized = optimizer.fit(skeleton, X, y)

        # Check coefficient
        assert np.abs(optimized.coef[0] - 5.0) < 0.01

    def test_linear_combination_multiple_vars(self):
        """Test linear combination of multiple variables."""
        skeleton = SkeletonExpression(
            name="linear",
            expression_str="a*x1 + b*x2 + c*x3 + d",
            n_input_vars=3,
        )

        # Generate data
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = 1.5 * X[:, 0] + 2.5 * X[:, 1] + 3.5 * X[:, 2] + 0.5

        # Optimize
        optimizer = SkeletonOptimizer()
        optimized = optimizer.fit(skeleton, X, y)

        # Check coefficients (sorted: a, b, c, d)
        assert np.abs(optimized.coef[0] - 1.5) < 0.1
        assert np.abs(optimized.coef[1] - 2.5) < 0.1
        assert np.abs(optimized.coef[2] - 3.5) < 0.1
        assert np.abs(optimized.coef[3] - 0.5) < 0.1

    def test_trigonometric_product(self):
        """Test skeleton with product of trigonometric functions."""
        skeleton = SkeletonExpression(
            name="trig_prod",
            expression_str="a*sin(x1)*cos(x1) + b",
            n_input_vars=1,
        )

        # Generate data: y = 4*sin(x)*cos(x) + 2 = 2*sin(2x) + 2
        np.random.seed(42)
        X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
        y = 4 * np.sin(X[:, 0]) * np.cos(X[:, 0]) + 2

        # Optimize
        optimizer = SkeletonOptimizer()
        optimized = optimizer.fit(skeleton, X, y)

        # Check coefficients
        assert np.abs(optimized.coef[0] - 4.0) < 0.1
        assert np.abs(optimized.coef[1] - 2.0) < 0.1


class TestSkeletonMultivariate:
    """Tests for multivariate skeleton expressions."""

    def test_linear_two_vars(self):
        """Test linear skeleton with two variables: y = 2*x1 + 3*x2 + 1."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 2 * X[:, 0] + 3 * X[:, 1] + 1

        skeleton = SkeletonExpression(
            name="lin2",
            expression_str="a*x1 + b*x2 + c",
            n_input_vars=2,
        )
        optimizer = SkeletonOptimizer()
        optimized = optimizer.fit(skeleton, X, y)

        assert np.abs(optimized.coef[0] - 2.0) < 0.1
        assert np.abs(optimized.coef[1] - 3.0) < 0.1
        assert np.abs(optimized.coef[2] - 1.0) < 0.1

        # Verify execution on new data
        X_test = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y_pred = optimized(X_test)
        y_expected = 2 * X_test[:, 0] + 3 * X_test[:, 1] + 1
        np.testing.assert_allclose(y_pred, y_expected, atol=0.2)

    def test_interaction_term(self):
        """Test skeleton with interaction x1*x2: y = 2*x1*x2 + 3*x1 + 1."""
        np.random.seed(42)
        X = np.random.randn(200, 2)
        y = 2 * X[:, 0] * X[:, 1] + 3 * X[:, 0] + 1

        skeleton = SkeletonExpression(
            name="interact",
            expression_str="a*x1*x2 + b*x1 + c",
            n_input_vars=2,
        )
        optimizer = SkeletonOptimizer()
        optimized = optimizer.fit(skeleton, X, y)

        assert np.abs(optimized.coef[0] - 2.0) < 0.1
        assert np.abs(optimized.coef[1] - 3.0) < 0.1
        assert np.abs(optimized.coef[2] - 1.0) < 0.1

    def test_fourier_two_vars(self):
        """Test Fourier skeleton across two variables."""
        np.random.seed(42)
        X = np.random.uniform(0, 2 * np.pi, (200, 2))
        y = 1.5 * np.cos(X[:, 0]) + 2.5 * np.sin(X[:, 1]) + 0.5

        skeleton = SkeletonExpression(
            name="fourier2",
            expression_str="a*cos(x1) + b*sin(x2) + c",
            n_input_vars=2,
        )
        optimizer = SkeletonOptimizer()
        optimized = optimizer.fit(skeleton, X, y)

        assert np.abs(optimized.coef[0] - 1.5) < 0.2
        assert np.abs(optimized.coef[1] - 2.5) < 0.2
        assert np.abs(optimized.coef[2] - 0.5) < 0.2

    def test_nonlinear_multivariate(self):
        """Test nonlinear skeleton with multiple variables."""
        np.random.seed(42)
        X = np.random.uniform(0.1, 3.0, (200, 2))
        # y = 2*exp(-0.5*(x1+x2)) + 1
        y = 2 * np.exp(-0.5 * (X[:, 0] + X[:, 1])) + 1

        skeleton = SkeletonExpression(
            name="exp_multi",
            expression_str="a*exp(-b*(x1 + x2)) + c",
            n_input_vars=2,
        )
        optimizer = SkeletonOptimizer(
            nonlinear_optimizer_params={
                "method": "L-BFGS-B",
                "options": {"maxiter": 500},
            }
        )
        optimized = optimizer.fit(skeleton, X, y)

        # Check predictions rather than exact coefficients (nonlinear)
        y_pred = optimized(X)
        mse = np.mean((y - y_pred) ** 2)
        assert mse < 0.1, f"MSE {mse} too high for multivariate nonlinear skeleton"

    def test_three_vars_linear(self):
        """Test linear skeleton with three variables."""
        np.random.seed(42)
        X = np.random.randn(200, 5)  # 5-feature dataset, skeleton uses 3
        y = 1.0 * X[:, 0] + 2.0 * X[:, 1] + 3.0 * X[:, 2] + 0.5

        skeleton = SkeletonExpression(
            name="lin3",
            expression_str="a*x1 + b*x2 + c*x3 + d",
            n_input_vars=5,  # dataset has 5 features, skeleton uses 3
        )
        optimizer = SkeletonOptimizer()
        optimized = optimizer.fit(skeleton, X, y)

        assert np.abs(optimized.coef[0] - 1.0) < 0.1
        assert np.abs(optimized.coef[1] - 2.0) < 0.1
        assert np.abs(optimized.coef[2] - 3.0) < 0.1
        assert np.abs(optimized.coef[3] - 0.5) < 0.1

    def test_subset_of_variables(self):
        """Test skeleton that uses x2 but not x1 in a 3-variable dataset."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = 4.0 * X[:, 1] + 2.0  # only uses x2

        skeleton = SkeletonExpression(
            name="x2_only",
            expression_str="a*x2 + b",
            n_input_vars=3,
        )
        optimizer = SkeletonOptimizer()
        optimized = optimizer.fit(skeleton, X, y)

        assert np.abs(optimized.coef[0] - 4.0) < 0.1
        assert np.abs(optimized.coef[1] - 2.0) < 0.1

        # Verify execution picks the right column
        X_test = np.array([[99.0, 1.0, 99.0], [99.0, 2.0, 99.0]])
        y_pred = optimized(X_test)
        y_expected = 4.0 * X_test[:, 1] + 2.0
        np.testing.assert_allclose(y_pred, y_expected, atol=0.2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
