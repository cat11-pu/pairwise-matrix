"""Behaviour tests for the dense matrix kernels.

Every expectation is written out as a plain result: a solution vector, a
determinant, a rank, a condition estimate or the exception a caller must see.
Run them from the project root:

    python3 -m unittest discover -s tests -v
"""

import unittest

from matrix.core import (
    MatrixError,
    back_substitution,
    condition_estimate,
    determinant,
    forward_substitution,
    identity,
    inverse,
    lu_decompose,
    lu_solve,
    matmul,
    rank,
    residual,
    transpose,
)


def packed_factors(result):
    """Split the packed array of a factorization into L and U."""
    n = len(result.lu)
    lower = [[1.0 if row == col else 0.0 for col in range(n)] for row in range(n)]
    upper = [[0.0] * n for _ in range(n)]
    for row in range(n):
        for col in range(n):
            if col < row:
                lower[row][col] = result.lu[row][col]
            else:
                upper[row][col] = result.lu[row][col]
    return lower, upper


def row_permuted(rows, perm):
    """The rows of a matrix reordered by a permutation."""
    return [list(rows[index]) for index in perm]


def applied(a, x):
    """A x for a matrix and a plain vector."""
    return [row[0] for row in matmul(a, [[value] for value in x])]


class MatrixKernelTest(unittest.TestCase):
    """Public behaviour of the matrix kernels."""

    def test_solves_square_systems(self):
        cases = (
            ([[2.0, 1.0], [1.0, 3.0]], [3.0, 5.0], [0.8, 1.4]),
            ([[1.0, 2.0], [3.0, 4.0]], [5.0, 11.0], [1.0, 2.0]),
            ([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]], [6.0, 10.0, 8.0], [1.0, 2.0, 3.0]),
            ([[-4.0, 1.0], [2.0, 3.0]], [-2.0, 8.0], [1.0, 2.0]),
        )
        for index, (a, b, expected) in enumerate(cases):
            x = lu_solve(a, b)
            self.assertEqual(len(x), len(expected), "case %d" % (index,))
            for row, (value, want) in enumerate(zip(x, expected)):
                self.assertAlmostEqual(value, want, places=9, msg="case %d entry %d" % (index, row))

    def test_solution_residual_is_small(self):
        a = [
            [6.0, 1.0, 1.0, 0.0],
            [1.0, 7.0, 2.0, 1.0],
            [0.0, 2.0, 8.0, 1.0],
            [1.0, 0.0, 1.0, 9.0],
        ]
        expected = [1.0, -2.0, 0.5, 3.0]
        b = applied(a, expected)
        x = lu_solve(a, b)
        self.assertLess(residual(a, x, b), 1e-9)
        for row, (value, want) in enumerate(zip(x, expected)):
            self.assertAlmostEqual(value, want, places=9, msg="entry %d" % (row,))
        self.assertLess(residual(a, expected, b), 1e-12)

    def test_factorization_rebuilds_the_matrix(self):
        matrices = (
            [[2.0, 1.0], [1.0, 3.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
        )
        for index, a in enumerate(matrices):
            result = lu_decompose(a)
            self.assertFalse(result.singular, "case %d" % (index,))
            self.assertEqual(sorted(result.perm), list(range(len(a))), "case %d" % (index,))
            lower, upper = packed_factors(result)
            product = matmul(lower, upper)
            rows = row_permuted(a, result.perm)
            for row in range(len(a)):
                for col in range(len(a)):
                    self.assertAlmostEqual(
                        product[row][col], rows[row][col], places=9, msg="case %d entry %d,%d" % (index, row, col)
                    )

    def test_forward_and_back_substitution(self):
        lower = [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, -1.0, 1.0]]
        y = forward_substitution(lower, [1.0, 0.0, 4.0])
        for row, (value, want) in enumerate(zip(y, [1.0, -2.0, 1.0])):
            self.assertAlmostEqual(value, want, places=9, msg="forward entry %d" % (row,))
        upper = [[2.0, 1.0, 1.0], [0.0, 3.0, 1.0], [0.0, 0.0, 4.0]]
        x = back_substitution(upper, [7.0, 9.0, 12.0])
        for row, (value, want) in enumerate(zip(x, [1.0, 2.0, 3.0])):
            self.assertAlmostEqual(value, want, places=9, msg="back entry %d" % (row,))

    def test_determinant_value_and_sign(self):
        self.assertAlmostEqual(determinant([[2.0, 1.0], [1.0, 3.0]]), 5.0, places=9)
        self.assertAlmostEqual(determinant([[0.0, 1.0], [1.0, 0.0]]), -1.0, places=9)
        self.assertAlmostEqual(
            determinant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]), -3.0, places=9
        )
        self.assertAlmostEqual(
            determinant([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 5.0]]), 30.0, places=9
        )

    def test_singular_and_near_singular_matrices_are_reported(self):
        self.assertTrue(lu_decompose([[1.0, 2.0], [2.0, 4.0]]).singular)
        with self.assertRaises(MatrixError):
            lu_solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])
        with self.assertRaises(MatrixError):
            inverse([[1.0, 2.0], [2.0, 4.0]])
        near = [[1.0, 1.0], [1.0, 1.0 + 1e-13]]
        self.assertTrue(lu_decompose(near, tol=1e-9).singular)
        with self.assertRaises(MatrixError):
            lu_solve(near, [1.0, 2.0], tol=1e-9)
        self.assertFalse(lu_decompose([[1.0, 2.0], [2.0, 5.0]], tol=1e-9).singular)

    def test_rank_counts_pivots_above_the_tolerance(self):
        self.assertEqual(rank([[1.0, 2.0], [3.0, 4.0]]), 2)
        self.assertEqual(rank([[0.0, 1.0], [-1.0, 0.0]]), 2)
        self.assertEqual(rank([[1.0, 2.0], [2.0, 4.0]]), 1)
        self.assertEqual(rank([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]]), 2)
        self.assertEqual(rank([[1.0, 1.0], [1.0, 1.0 + 1e-13]], tol=1e-9), 1)

    def test_inverse_undoes_the_matrix(self):
        a = [[4.0, 3.0], [6.0, 5.0]]
        inv = inverse(a)
        expected = [[2.5, -1.5], [-3.0, 2.0]]
        for row in range(2):
            for col in range(2):
                self.assertAlmostEqual(inv[row][col], expected[row][col], places=9, msg="entry %d,%d" % (row, col))
        product = matmul(a, inv)
        for row in range(2):
            for col in range(2):
                self.assertAlmostEqual(
                    product[row][col], identity(2)[row][col], places=9, msg="product %d,%d" % (row, col)
                )
        three = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
        round_trip = matmul(three, inverse(three))
        for row in range(3):
            for col in range(3):
                self.assertAlmostEqual(
                    round_trip[row][col], identity(3)[row][col], places=9, msg="round trip %d,%d" % (row, col)
                )

    def test_condition_estimate_grows_with_the_pivots(self):
        self.assertEqual(condition_estimate([[1.0, 2.0], [2.0, 4.0]]), float("inf"))
        self.assertAlmostEqual(condition_estimate(identity(3)), 1.0, places=9)
        self.assertAlmostEqual(condition_estimate([[1.0, 2.0], [3.0, 4.0]]), 4.5, places=9)
        self.assertGreaterEqual(
            condition_estimate([[1.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1000.0]]), 1.0
        )

    def test_rejects_malformed_input(self):
        with self.assertRaises(MatrixError):
            lu_decompose([[1.0, 2.0], [3.0]])
        with self.assertRaises(MatrixError):
            lu_decompose([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with self.assertRaises(MatrixError):
            lu_decompose([[1.0, 2.0], [3.0, 4.0]], tol=-1.0)
        with self.assertRaises(MatrixError):
            lu_solve([[1.0, 0.0], [0.0, 1.0]], [1.0])
        with self.assertRaises(MatrixError):
            forward_substitution([[1.0, 0.0], [0.0, 1.0]], [1.0])
        with self.assertRaises(MatrixError):
            identity(0)
        with self.assertRaises(MatrixError):
            matmul([[1.0, 2.0]], [[1.0, 2.0]])
        self.assertEqual(transpose([[1.0, 2.0], [3.0, 4.0]]), [[1.0, 3.0], [2.0, 4.0]])
        self.assertEqual(transpose(transpose([[1.0, 2.0], [3.0, 4.0]])), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(residual([[1.0, 0.0], [0.0, 1.0]], [3.0, 4.0], [3.0, 4.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
