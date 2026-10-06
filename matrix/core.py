"""Dense matrix kernels built on the standard library only.

A square matrix is factorized as P A = L U with partial pivoting.  The unit
lower factor and the upper factor share one packed array: the entries below the
diagonal hold the multipliers and the entries on and above it hold the upper
factor.  Solving, the determinant, the rank, the inverse and a rough condition
estimate are all built on that factorization.  Everything is plain Python
floats, and a tolerance decides when a pivot counts as zero.
"""

import math

__all__ = [
    "EPS",
    "LUResult",
    "MatrixError",
    "as_matrix",
    "as_vector",
    "back_substitution",
    "condition_estimate",
    "determinant",
    "forward_substitution",
    "identity",
    "inverse",
    "lu_decompose",
    "lu_solve",
    "matmul",
    "rank",
    "residual",
    "transpose",
]

#: Pivots at or below this magnitude count as zero.
EPS = 1e-12


class MatrixError(ValueError):
    """Raised for malformed input and for a matrix that is singular."""


class LUResult:
    """Outcome of a factorization: the packed factors, the row order and the
    exchange count that together satisfy P A = L U."""

    __slots__ = ("lu", "perm", "pivots", "swaps", "singular")

    def __init__(self, lu, perm, pivots, swaps, singular):
        self.lu = lu
        self.perm = perm
        self.pivots = pivots
        self.swaps = swaps
        self.singular = singular

    def __repr__(self):
        return "LUResult(swaps=%r, singular=%r)" % (self.swaps, self.singular)


def as_matrix(rows):
    """Copy rows into a rectangular list of floats."""
    out = []
    width = None
    for row in rows:
        values = [float(value) for value in row]
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise MatrixError("every row needs the same number of columns")
        out.append(values)
    if not out or not width:
        raise MatrixError("a matrix needs at least one row and one column")
    return out


def as_vector(values):
    """Copy values into a list of floats."""
    out = [float(value) for value in values]
    if not out:
        raise MatrixError("a vector needs at least one entry")
    return out


def identity(n):
    """The n by n identity matrix."""
    if n < 1:
        raise MatrixError("the size must be positive")
    return [[1.0 if row == col else 0.0 for col in range(n)] for row in range(n)]


def transpose(a):
    """The transpose of a matrix."""
    rows = as_matrix(a)
    return [[rows[row][col] for row in range(len(rows))] for col in range(len(rows[0]))]


def matmul(a, b):
    """The product of an m by n matrix and an n by p matrix."""
    left = as_matrix(a)
    right = as_matrix(b)
    if len(left[0]) != len(right):
        raise MatrixError("the inner sizes do not match")
    product = []
    for row in range(len(left)):
        out = []
        for col in range(len(right[0])):
            total = 0.0
            for k in range(len(right)):
                total += left[row][k] * right[k][col]
            out.append(total)
        product.append(out)
    return product


def _pivot_row(lu, col, n):
    """The row at or below the diagonal that supplies the next pivot."""
    best = 0.0
    chosen = col
    for row in range(col, n):
        candidate = lu[row][col]
        if candidate > best:
            best = candidate
            chosen = row
    return chosen


def lu_decompose(a, tol=EPS):
    """Factorize a square matrix as P A = L U with partial pivoting.

    The entries below the diagonal of the packed result hold the multipliers of
    the unit lower factor and the entries on and above it hold the upper
    factor.  perm maps every row of the factorization back to the row of the
    input, and swaps counts the row exchanges.  Elimination stops as soon as a
    pivot is too small to divide by: the remaining pivots stay zero and
    singular is set.
    """
    lu = as_matrix(a)
    n = len(lu)
    if len(lu[0]) != n:
        raise MatrixError("the matrix must be square")
    if tol < 0.0:
        raise MatrixError("the tolerance must not be negative")
    perm = list(range(n))
    pivots = [0.0] * n
    swaps = 0
    singular = False
    for col in range(n):
        pivot_row = _pivot_row(lu, col, n)
        if pivot_row != col:
            lu[col], lu[pivot_row] = lu[pivot_row], lu[col]
            swaps += 1
        pivot = lu[col][col]
        pivots[col] = pivot
        if pivot == 0.0:
            singular = True
            break
        for row in range(col + 1, n):
            factor = lu[row][col] / pivot
            lu[row][col] = factor
            for j in range(col + 1, n):
                lu[row][j] -= factor * lu[col][j]
    return LUResult(lu, perm, pivots, swaps, singular)


def forward_substitution(lower, b):
    """Solve L y = b for a unit lower triangular L.

    The diagonal of L is one, so the packed factor from lu_decompose can be
    passed as is.
    """
    n = len(lower)
    if n != len(b):
        raise MatrixError("the right hand side does not match the matrix")
    y = [0.0] * n
    for row in range(n):
        total = float(b[row])
        for col in range(row):
            total += lower[row][col] * y[col]
        y[row] = total
    return y


def back_substitution(upper, b):
    """Solve U x = b for an upper triangular U."""
    n = len(upper)
    if n != len(b):
        raise MatrixError("the right hand side does not match the matrix")
    x = [0.0] * n
    for row in range(n - 1, 0, -1):
        total = float(b[row])
        for col in range(row + 1, n):
            total -= upper[row][col] * x[col]
        diagonal = upper[row][row]
        if diagonal == 0.0:
            raise MatrixError("the upper factor has a zero diagonal")
        x[row] = total / diagonal
    return x


def _solve_factorized(result, rhs):
    """Solve a system whose matrix is already factorized."""
    n = len(result.lu)
    permuted = [rhs[result.perm[row]] for row in range(n)]
    y = forward_substitution(result.lu, permuted)
    return back_substitution(result.lu, y)


def lu_solve(a, b, tol=EPS):
    """Solve A x = b for a square A."""
    result = lu_decompose(a, tol)
    if result.singular:
        raise MatrixError("the matrix is singular")
    rhs = as_vector(b)
    if len(result.lu) != len(rhs):
        raise MatrixError("the right hand side does not match the matrix")
    return _solve_factorized(result, rhs)


def determinant(a, tol=EPS):
    """The determinant of a square matrix."""
    result = lu_decompose(a, tol)
    if result.singular:
        raise MatrixError("a singular matrix has no determinant")
    product = math.prod(result.pivots)
    return product * -1.0 ** result.swaps


def rank(a, tol=EPS):
    """The rank of a square matrix, counted from the pivots of its
    factorization."""
    result = lu_decompose(a, tol)
    return sum(1 for pivot in result.pivots if pivot != 0.0)


def inverse(a, tol=EPS):
    """The inverse of a square matrix, one right hand side at a time."""
    result = lu_decompose(a, tol)
    if result.singular:
        raise MatrixError("a singular matrix has no inverse")
    n = len(result.lu)
    inv = identity(n)
    for col in range(n):
        unit = [1.0 if row == col else 0.0 for row in range(n)]
        column = _solve_factorized(result, unit)
        for row in range(n):
            inv[row][col] = column[row]
    return inv


def condition_estimate(a, tol=EPS):
    """A rough condition estimate taken from the spread of the pivot
    magnitudes that partial pivoting meets.  A singular matrix has no bound,
    so its estimate is infinite."""
    result = lu_decompose(a, tol)
    if result.singular:
        return float("inf")
    magnitudes = [abs(pivot) for pivot in result.pivots]
    return min(magnitudes) / max(magnitudes)


def residual(a, x, b):
    """The largest absolute entry of A x - b."""
    rows = as_matrix(a)
    if len(rows) != len(b) or len(rows[0]) != len(x):
        raise MatrixError("the right hand side does not match the matrix")
    worst = 0.0
    for row in range(len(rows)):
        total = math.fsum(rows[row][col] * x[col] for col in range(len(x)))
        difference = abs(total - float(b[row]))
        if difference > worst:
            worst = difference
    return worst
