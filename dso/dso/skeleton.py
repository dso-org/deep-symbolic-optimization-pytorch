"""Skeleton expression optimizer for deep symbolic optimization.

Skeleton expressions generalize the polynomial token to arbitrary mathematical
templates with optimizable coefficients. For example, "a*cos(x1) + b*sin(x1) + c"
where a, b, c are coefficients optimized to fit data.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import sympy as sp
from scipy.optimize import minimize

from dso.library import Token
from dso.task.regression.polyfit import PolyRegressorMixin, regressors


def parse_skeleton(
    expression_str: str,
    n_input_vars: int,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse skeleton expression string into structured metadata.

    Parameters
    ----------
    expression_str : str
        Mathematical expression like "a*cos(x1) + b*sin(x1) + c"
    n_input_vars : int
        Number of input variables in dataset
    name : str, optional
        Name for the skeleton token

    Returns
    -------
    metadata : dict
        Dictionary containing:
        - sympy_expr: SymPy expression object
        - coef_names: List of coefficient symbol names (sorted)
        - var_names: List of variable symbol names
        - coef_symbols: List of SymPy coefficient symbols
        - var_symbols: List of SymPy variable symbols
        - is_linear: Whether expression is linear in coefficients
        - n_coefs: Number of coefficients
        - basis_funcs: List of compiled basis functions (for linear case)
        - eval_func: Compiled evaluation function (for nonlinear case)

    Raises
    ------
    ValueError
        If expression is invalid or uses undefined variables
    """
    try:
        sympy_expr = sp.sympify(expression_str)
    except Exception as e:
        raise ValueError(f"Failed to parse skeleton expression '{expression_str}': {e}")
    if not isinstance(sympy_expr, sp.Basic):
        raise ValueError(
            f"Skeleton expression '{expression_str}' parsed to unsupported type "
            f"'{type(sympy_expr).__name__}'. Use a symbolic formula like "
            "'a*x1**2 + b*x1 + c', not a bare token name like 'poly'."
        )

    # Get all free symbols
    free_symbols = sympy_expr.free_symbols
    if not free_symbols:
        raise ValueError(
            f"Skeleton expression '{expression_str}' has no free symbols. "
            "It must contain at least one coefficient or variable."
        )

    # Identify variable symbols (x1, x2, ..., xN)
    var_pattern = {sp.Symbol(f"x{i+1}") for i in range(n_input_vars)}
    var_symbols = sorted(
        [s for s in free_symbols if s in var_pattern],
        key=lambda s: int(str(s)[1:]),  # Sort by number
    )
    var_names = [str(s) for s in var_symbols]

    if invalid_vars := [
        s
        for s in free_symbols
        if str(s).startswith("x") and s not in var_pattern
    ]:
        raise ValueError(
            f"Skeleton expression uses invalid variables {invalid_vars}. "
            f"Valid variables are x1 through x{n_input_vars}."
        )

    # Remaining symbols are coefficients
    coef_symbols = sorted(
        [s for s in free_symbols if s not in var_pattern],
        key=str,  # Sort alphabetically
    )
    coef_names = [str(s) for s in coef_symbols]

    if not coef_symbols:
        raise ValueError(
            f"Skeleton expression '{expression_str}' has no coefficients to optimize. "
            "All free symbols are variables."
        )

    # Detect if expression is linear in coefficients
    is_linear = _is_linear_in_coefs(sympy_expr, coef_symbols)

    metadata = {
        "sympy_expr": sympy_expr,
        "coef_names": coef_names,
        "var_names": var_names,
        "coef_symbols": coef_symbols,
        "var_symbols": var_symbols,
        "is_linear": is_linear,
        "n_coefs": len(coef_symbols),
    }

    # Compile functions for execution
    if is_linear:
        # For linear expressions, compile each basis function
        metadata["basis_funcs"] = _compile_basis_functions(
            sympy_expr, coef_symbols, var_symbols
        )
    else:
        # For nonlinear expressions, compile full evaluation function
        metadata["eval_func"] = _compile_eval_function(
            sympy_expr, coef_symbols, var_symbols
        )

    return metadata


def _is_linear_in_coefs(expr: sp.Expr, coef_symbols: List[sp.Symbol]) -> bool:
    """Check if expression is linear in all coefficient symbols.

    An expression is linear in coefficients if the Hessian matrix with
    respect to coefficients is zero everywhere.
    """
    if not coef_symbols:
        return True

    # Check if each coefficient appears at most linearly
    for coef in coef_symbols:
        # Take first derivative
        first_deriv = expr.diff(coef)

        # Check if first derivative contains any coefficient
        # (which would mean the coefficient appears nonlinearly)
        if any(c in first_deriv.free_symbols for c in coef_symbols):
            return False

    return True


def _compile_basis_functions(
    expr: sp.Expr,
    coef_symbols: List[sp.Symbol],
    var_symbols: List[sp.Symbol],
) -> List[Callable]:
    """Compile basis functions for linear skeleton expressions.

    For a linear expression like a*f1(x) + b*f2(x) + c*f3(x),
    returns [f1, f2, f3] as compiled numpy functions.
    """
    basis_funcs = []

    for coef in coef_symbols:
        # Extract the basis function by taking derivative w.r.t. coefficient
        # For linear expressions, this gives us the basis function
        basis_expr = expr.diff(coef)

        # Lambdify to numpy function
        # var_symbols are the inputs (x1, x2, ...)
        if var_symbols:
            basis_func = sp.lambdify(var_symbols, basis_expr, modules="numpy")
        else:
            # Constant term
            basis_func = lambda *args: np.ones(1) * float(basis_expr)

        basis_funcs.append(basis_func)

    return basis_funcs


def _compile_eval_function(
    expr: sp.Expr,
    coef_symbols: List[sp.Symbol],
    var_symbols: List[sp.Symbol],
) -> Callable:
    """Compile full evaluation function for nonlinear skeleton expressions.

    Returns a function that takes (coef_values, X) and evaluates the expression.
    """
    all_symbols = coef_symbols + var_symbols
    return sp.lambdify(all_symbols, expr, modules="numpy")


class SkeletonExpression(Token):
    """Token representing a parameterized expression template.

    Skeleton expressions allow users to specify mathematical templates like
    "a*cos(x1) + b*sin(x1) + c" where coefficients (a, b, c) are optimized
    to fit data.

    Parameters
    ----------
    name : str
        Unique identifier for this skeleton token
    expression_str : str
        Mathematical expression with coefficients and variables
    n_input_vars : int
        Number of input variables in the dataset
    coef : np.ndarray, optional
        Optimized coefficient values (None until optimized)
    metadata : dict, optional
        Parsed skeleton metadata (computed if not provided)

    Attributes
    ----------
    expression_str : str
        Original expression string
    metadata : dict
        Parsed expression metadata (coef_names, is_linear, etc.)
    coef : np.ndarray or None
        Optimized coefficient values
    n_input_vars : int
        Number of input variables
    """

    def __init__(
        self,
        name: str,
        expression_str: str,
        n_input_vars: int,
        coef: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_count: int = 1,
    ):
        self.name = name
        self.expression_str = expression_str
        self.n_input_vars = n_input_vars
        self.coef = coef
        self.max_count = max_count

        # Parse expression if metadata not provided
        if metadata is None:
            metadata = parse_skeleton(expression_str, n_input_vars, name)
        self.metadata = metadata

        # Set complexity: 1 before optimization, len(coef) after
        complexity = 1 if coef is None else len(coef)

        # Initialize Token with eval function
        super().__init__(
            function=self._eval_skeleton,
            name=name,
            arity=0,  # Terminal token, takes no children
            complexity=complexity,
        )

    def _eval_skeleton(self, X: np.ndarray) -> np.ndarray:
        """Evaluate skeleton expression on input data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data

        Returns
        -------
        y : np.ndarray, shape (n_samples,)
            Evaluated expression
        """
        if self.coef is None:
            raise ValueError(
                f"Skeleton '{self.name}' has not been optimized. "
                "Coefficients are None."
            )

        n_samples = X.shape[0]

        if self.metadata["is_linear"]:
            # Linear case: evaluate basis functions and combine
            return self._eval_linear(X)
        else:
            # Nonlinear case: evaluate full expression
            return self._eval_nonlinear(X)

    def _eval_linear(self, X: np.ndarray) -> np.ndarray:
        """Evaluate linear skeleton expression."""
        basis_funcs = self.metadata["basis_funcs"]
        var_symbols = self.metadata["var_symbols"]
        n_samples = X.shape[0]

        # Create design matrix by evaluating each basis function
        design_matrix = np.zeros((n_samples, len(basis_funcs)))

        for i, basis_func in enumerate(basis_funcs):
            if var_symbols:
                # Extract variable columns
                var_indices = [
                    int(str(s)[1:]) - 1 for s in var_symbols
                ]  # x1 -> index 0
                X_vars = X[:, var_indices]

                # Call basis function with unpacked variables
                if len(var_symbols) == 1:
                    result = basis_func(X_vars[:, 0])
                else:
                    result = basis_func(*[X_vars[:, j] for j in range(len(var_symbols))])
            else:
                # Constant basis
                result = basis_func()

            # Ensure result is array and has correct shape
            result = np.atleast_1d(result)
            if result.shape[0] == 1:
                result = np.full(n_samples, result[0])

            design_matrix[:, i] = result.ravel()

        # Combine basis functions with coefficients
        return design_matrix @ self.coef

    def _eval_nonlinear(self, X: np.ndarray) -> np.ndarray:
        """Evaluate nonlinear skeleton expression."""
        eval_func = self.metadata["eval_func"]
        n_samples = X.shape[0]

        if var_symbols := self.metadata["var_symbols"]:
            var_indices = [int(str(s)[1:]) - 1 for s in var_symbols]
            X_vars = X[:, var_indices]

            # Call eval function with coefficients and variables
            args = list(self.coef) + [X_vars[:, j] for j in range(len(var_symbols))]
        else:
            # Only coefficients, no variables
            args = list(self.coef)

        result = eval_func(*args)

        # Ensure result is array with correct shape
        result = np.atleast_1d(result)
        if result.shape[0] == 1:
            result = np.full(n_samples, result[0])

        return result.ravel()

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Call interface for skeleton evaluation."""
        return self._eval_skeleton(X)

    def __repr__(self):
        """String representation of skeleton token."""
        if self.coef is None:
            return f"{self.name}[{self.expression_str}]"

        # Show expression with optimized coefficients
        coef_dict = dict(zip(self.metadata["coef_names"], self.coef))

        # Try to create a nice string representation
        expr_str = self.expression_str
        for coef_name, coef_val in coef_dict.items():
            # Replace coefficient with value
            expr_str = expr_str.replace(coef_name, f"{coef_val:.4f}")

        return f"{self.name}[{expr_str}]"


class SkeletonOptimizer(PolyRegressorMixin):
    """Optimizer for skeleton expression coefficients.

    Automatically detects if skeleton is linear (uses least squares) or
    nonlinear (uses scipy.minimize) and applies appropriate optimization.

    Parameters
    ----------
    coef_tol : float
        Cutoff for zero coefficients (magnitude below this set to zero)
    linear_regressor : str
        Regressor name for linear case ("dso_least_squares", "lasso", etc.)
    linear_regressor_params : dict
        Parameters for linear regressor
    nonlinear_optimizer : str
        Optimizer for nonlinear case ("scipy")
    nonlinear_optimizer_params : dict
        Parameters for scipy.minimize
    """

    def __init__(
        self,
        coef_tol: float = 1e-6,
        linear_regressor: str = "dso_least_squares",
        linear_regressor_params: Optional[Dict] = None,
        nonlinear_optimizer: str = "scipy",
        nonlinear_optimizer_params: Optional[Dict] = None,
    ):
        self.coef_tol = coef_tol

        # Setup linear regressor
        if linear_regressor_params is None:
            linear_regressor_params = {}
        self.linear_regressor = regressors[linear_regressor](**linear_regressor_params)

        # Setup nonlinear optimizer parameters
        if nonlinear_optimizer_params is None:
            nonlinear_optimizer_params = {
                "method": "L-BFGS-B",
                "options": {"maxiter": 100},
            }
        self.nonlinear_optimizer = nonlinear_optimizer
        self.nonlinear_optimizer_params = nonlinear_optimizer_params

        # Cache for basis function evaluations (linear case)
        self.data_dict = {}
        self.n_max_records = 10

    def fit(
        self,
        skeleton: SkeletonExpression,
        X: np.ndarray,
        y: np.ndarray,
    ) -> SkeletonExpression:
        """Optimize skeleton coefficients to fit data.

        Parameters
        ----------
        skeleton : SkeletonExpression
            Skeleton token to optimize (with coef=None)
        X : np.ndarray, shape (n_samples, n_features)
            Input data
        y : np.ndarray, shape (n_samples,)
            Target values

        Returns
        -------
        optimized_skeleton : SkeletonExpression
            New skeleton with optimized coefficients
        """
        if skeleton.metadata["is_linear"]:
            return self._fit_linear(skeleton, X, y)
        else:
            return self._fit_nonlinear(skeleton, X, y)

    def _fit_linear(
        self,
        skeleton: SkeletonExpression,
        X: np.ndarray,
        y: np.ndarray,
    ) -> SkeletonExpression:
        """Fit linear skeleton using least squares regression."""
        metadata = skeleton.metadata
        basis_funcs = metadata["basis_funcs"]
        var_symbols = metadata["var_symbols"]
        n_samples = X.shape[0]

        # Build design matrix by evaluating each basis function
        design_matrix = np.zeros((n_samples, len(basis_funcs)))

        for i, basis_func in enumerate(basis_funcs):
            if var_symbols:
                var_indices = [int(str(s)[1:]) - 1 for s in var_symbols]
                X_vars = X[:, var_indices]

                if len(var_symbols) == 1:
                    result = basis_func(X_vars[:, 0])
                else:
                    result = basis_func(*[X_vars[:, j] for j in range(len(var_symbols))])
            else:
                result = basis_func()

            result = np.atleast_1d(result)
            if result.shape[0] == 1:
                result = np.full(n_samples, result[0])

            design_matrix[:, i] = result.ravel()

        # Fit using regressor
        X_signature = self.np_array_signature(design_matrix)
        self.linear_regressor.fit(design_matrix, y, X_signature)

        # Get coefficients and filter by tolerance
        coef = self.linear_regressor.coef_
        mask = np.abs(coef) >= self.coef_tol

        # Keep only significant coefficients
        if not np.any(mask):
            # All coefficients filtered out, keep the largest one
            mask[np.argmax(np.abs(coef))] = True

        significant_coef = coef[mask]

        # For now, we keep all coefficients (even if some are small)
        # This matches the poly token behavior more closely
        # In future, could filter basis functions based on mask
        optimized_coef = coef

        # Create new skeleton with optimized coefficients
        return SkeletonExpression(
            name=skeleton.name,
            expression_str=skeleton.expression_str,
            n_input_vars=skeleton.n_input_vars,
            coef=optimized_coef,
            metadata=metadata,
        )

    def _fit_nonlinear(
        self,
        skeleton: SkeletonExpression,
        X: np.ndarray,
        y: np.ndarray,
    ) -> SkeletonExpression:
        """Fit nonlinear skeleton using scipy.minimize."""
        metadata = skeleton.metadata
        eval_func = metadata["eval_func"]
        var_symbols = metadata["var_symbols"]
        n_coefs = metadata["n_coefs"]

        # Prepare variable data
        if var_symbols:
            var_indices = [int(str(s)[1:]) - 1 for s in var_symbols]
            X_vars = X[:, var_indices]
        else:
            X_vars = None

        # Define objective function (MSE)
        def objective(coef_values):
            try:
                # Evaluate skeleton with current coefficients
                if X_vars is not None:
                    args = list(coef_values) + [
                        X_vars[:, j] for j in range(len(var_symbols))
                    ]
                else:
                    args = list(coef_values)

                y_pred = eval_func(*args)
                y_pred = np.atleast_1d(y_pred)

                # Ensure broadcasting works
                if y_pred.shape[0] == 1:
                    y_pred = np.full_like(y, y_pred[0])

                # Compute MSE
                mse = np.mean((y - y_pred.ravel()) ** 2)

                # Add penalty for very large coefficients
                penalty = 0.0001 * np.sum(coef_values**2)

                return mse + penalty
            except Exception:
                # Return large value if evaluation fails
                return 1e10

        # Initial guess: all ones (or zeros for offset terms)
        x0 = np.ones(n_coefs)

        # Set reasonable bounds to prevent numerical overflow
        bounds = [(-1e3, 1e3) for _ in range(n_coefs)]

        # Optimize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                objective,
                x0,
                bounds=bounds,
                **self.nonlinear_optimizer_params,
            )

        optimized_coef = result.x

        # Filter very small coefficients
        mask = np.abs(optimized_coef) >= self.coef_tol
        if not np.any(mask):
            mask[np.argmax(np.abs(optimized_coef))] = True

        # For now, keep all coefficients
        optimized_coef = result.x

        # Create new skeleton with optimized coefficients
        return SkeletonExpression(
            name=skeleton.name,
            expression_str=skeleton.expression_str,
            n_input_vars=skeleton.n_input_vars,
            coef=optimized_coef,
            metadata=metadata,
        )
