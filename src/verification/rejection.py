"""Exact accelerated verifier for Algorithm S2 at h=3 (physical T powers).

The rejection sets are exactly the current manuscript's sets, not the older
stronger modulo-four test. This module never changes code-instance data.

Public APIs:
  verify_current_algorithm(u, a, k, d) -> JSON-serializable certificate
  kernel_decomposition(matrix, s, ncols=None) -> KernelDecomposition
  primitive_kernel(matrix, s, ncols=None) -> witness vector or None

The matrix routines use exact integer arithmetic. NumPy is used only to form
the normalizer matrix and restrict columns. Search storage is polynomial apart
from the usual binary meet-in-the-middle table at the requested fixed distance.
Run ``python construct.py self-test`` for independent literal cross-checks,
or ``python construct.py verify`` to check the stored construction inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import random
from typing import Iterator

import numpy as np

from core import ring8 as _literal

# Scaled pairs (alpha,beta): the actual modulo-eight pair is (2alpha,2beta).
S2_COST_ONE = {(0, 1), (0, 2), (0, 3), (2, 0)} | {
    (a, b) for a in (1, 3) for b in (0, 1, 3)
}


def _valuation(x: int, s: int) -> int:
    x %= 1 << s
    return (x & -x).bit_length() - 1 if x else s


@dataclass
class KernelDecomposition:
    """A diagonal reduction P*A*V over Z/(2^s), with P,V invertible.

    Diagonal entries are 2^valuation; valuation s denotes a zero column.
    The kernel parameter on column j has order 2^valuation[j] and multiplier
    2^(s-valuation[j]). Consequently a primitive kernel exists exactly when
    one valuation equals s. Invertible V preserves primitivity.
    """

    s: int
    valuations: list[int]
    columns: list[list[int]]

    def at_modulus(self, s: int) -> "KernelDecomposition":
        if not 1 <= s <= self.s:
            raise ValueError("Can only reduce to a smaller positive exponent")
        modulus = 1 << s
        return KernelDecomposition(s, [min(v, s) for v in self.valuations],
                                   [[x % modulus for x in col] for col in self.columns])

    def primitive_witness(self) -> list[int] | None:
        for v, col in zip(self.valuations, self.columns):
            if v == self.s:
                return col[:]
        return None

    def generators(self) -> list[tuple[int, list[int]]]:
        modulus = 1 << self.s
        return [(1 << v, [(x << (self.s-v)) % modulus for x in col])
                for v, col in zip(self.valuations, self.columns) if v]

    def vectors(self, primitive_only: bool = False) -> Iterator[list[int]]:
        """Enumerate each kernel vector once from its invariant-factor digits."""
        modulus = 1 << self.s
        generators = self.generators()
        for coefficients in product(*(range(order) for order, _ in generators)):
            word = [0] * len(self.columns)
            for coefficient, (_, generator) in zip(coefficients, generators):
                if coefficient:
                    for j, value in enumerate(generator):
                        word[j] = (word[j] + coefficient * value) % modulus
            if not primitive_only or any(x & 1 for x in word):
                yield word


def kernel_decomposition(matrix, s: int, ncols: int | None = None) -> KernelDecomposition:
    """Exact Smith-style elimination; column transformations are retained.

    The least-valuation pivot divides every remaining entry. Unit row scaling,
    row subtraction and column subtraction isolate that pivot. These invertible
    operations preserve the complete kernel, including non-free ring torsion.
    """
    if s < 1:
        raise ValueError("s must be positive")
    modulus = 1 << s
    z = [[int(x) % modulus for x in row] for row in matrix]
    nc = len(z[0]) if z else (0 if ncols is None else ncols)
    if ncols is not None and nc != ncols:
        raise ValueError("ncols disagrees with matrix")
    if any(len(row) != nc for row in z):
        raise ValueError("Ragged matrix")
    nr = len(z)
    transform = [[int(i == j) for j in range(nc)] for i in range(nc)]
    valuations = [s] * nc
    pivot = 0
    while pivot < min(nr, nc):
        choice = min(((_valuation(z[i][j], s), i, j)
                      for i in range(pivot, nr) for j in range(pivot, nc)), default=(s, 0, 0))
        val, row, col = choice
        if val == s:
            break
        z[pivot], z[row] = z[row], z[pivot]
        if col != pivot:
            for line in z:
                line[pivot], line[col] = line[col], line[pivot]
            for line in transform:
                line[pivot], line[col] = line[col], line[pivot]
        unit_inverse = pow(z[pivot][pivot] >> val, -1, modulus)
        z[pivot] = [x * unit_inverse % modulus for x in z[pivot]]
        for i in range(pivot + 1, nr):
            coefficient = z[i][pivot] >> val
            if coefficient:
                z[i] = [(x - coefficient*y) % modulus for x, y in zip(z[i], z[pivot])]
        for j in range(pivot + 1, nc):
            coefficient = z[pivot][j] >> val
            if coefficient:
                for line in z:
                    line[j] = (line[j] - coefficient*line[pivot]) % modulus
                for line in transform:
                    line[j] = (line[j] - coefficient*line[pivot]) % modulus
        valuations[pivot] = val
        pivot += 1
    return KernelDecomposition(s, valuations,
                               [[transform[i][j] for i in range(nc)] for j in range(nc)])


def primitive_kernel(matrix, s: int, ncols: int | None = None) -> list[int] | None:
    return kernel_decomposition(matrix, s, ncols).primitive_witness()


def _independent_binary_columns(columns) -> bool:
    pivots = {}
    for column in columns:
        value = int(column)
        while value:
            pivot = value.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = value
                break
            value ^= pivots[pivot]
        if not value:
            return False
    return True


def _binary_ball(columns, radius):
    yield 0, 0
    for weight in range(1, radius + 1):
        for support in combinations(range(len(columns)), weight):
            syndrome = mask = 0
            for j in support:
                syndrome ^= columns[j]
                mask |= 1 << j
            yield syndrome, mask


def _s1_certificate(normal, inverse, k, d):
    r, twice_m = normal.shape
    m = twice_m // 2
    columns = [sum((int(normal[i, j]) & 1) << i for i in range(r))
               for j in range(twice_m)]
    for j in range(m):
        columns[j] |= sum((inverse[j][r+ell] & 1) << (r+ell) for ell in range(k))
    radius = (d-1) // 2
    seen = {}
    checked = 0
    for syndrome, mask in _binary_ball(columns, radius):
        checked += 1
        old = seen.get(syndrome)
        if old is not None and old != mask:
            return old ^ mask, checked
        seen[syndrome] = mask
    if (d-1) & 1:
        # The larger side is needed only when the maximum forbidden weight is odd.
        for syndrome, mask in _binary_ball(columns, radius + 1):
            checked += 1
            old = seen.get(syndrome)
            if old is not None and old != mask:
                return old ^ mask, checked
    return None, checked


def s2_cost(word) -> int:
    width = len(word) // 2
    return sum(0 if not (a or b) else 1 if (a, b) in S2_COST_ONE else 2
               for a, b in zip(word[:width], word[width:]))


def _embed_witness(word, support, m, s):
    multiplier = 1 << (3-s)
    width = len(support)
    a, b = [0] * m, [0] * m
    for index, position in enumerate(support):
        a[position] = multiplier * word[index]
        b[position] = multiplier * word[width+index]
    return {"s": s, "a": a, "b": b}


def _check_witness(u, matrix, inverse, k, d, witness):
    s, a, b = witness["s"], witness["a"], witness["b"]
    m, r = len(u), len(u)-k
    assert min(_valuation(x, 3) for x in a+b) == 3-s
    if s == 1:
        assert sum(bool(x) for x in a+b) < d
        assert not any(_literal.row_product(a, inverse)[r:])
    elif s == 2:
        assert s2_cost([x//2 for x in a+b]) < d
    else:
        assert sum(bool(x or y) for x, y in zip(a,b)) < d
    residual = [(x-y) % 8 for x,y in zip(b, _literal.row_product(a,matrix))]
    assert all(sum(x*y for x,y in zip(g,residual)) % 8 == 0 for g in u[:r])


def verify_current_algorithm(u, a, k: int, d: int) -> dict:
    """Accept iff Algorithm S2 accepts this h=3 candidate.

    For each maximal carrier support, binary independence rules out primitive
    kernels at both higher orders. Otherwise a tracked reduction modulo eight
    gives the modulo-four kernel too. Enumerating its primitive vectors with
    the exact local cost implements s=2; a free modulo-eight kernel coordinate
    implements s=3. Every smaller support is contained in a tested maximal one.
    """
    m = len(u)
    if not (1 <= k < m and 1 <= d <= m):
        raise ValueError("Require 1 <= K < M and 1 <= d <= M")
    for matrix in (u, a):
        if len(matrix) != m or any(len(row) != m for row in matrix):
            raise ValueError("Matrices must be square")
        if any(not isinstance(x, (int, np.integer)) or not 0 <= x < 8
               for row in matrix for x in row):
            raise ValueError("Entries must be integer representatives 0..7")
    u, a = [[int(x) for x in row] for row in u], [[int(x) for x in row] for row in a]
    if any(a[i][j] != a[j][i] for i in range(m) for j in range(m)):
        raise ValueError("A must be symmetric")
    inverse = _literal.inverse_mod8(u)
    if inverse is None:
        return {"passed": False, "reason": "U is singular modulo two"}
    g = np.asarray(u[:m-k], dtype=np.int64)
    normal = np.concatenate((-g @ np.asarray(a, dtype=np.int64), g), axis=1) % 8
    mask, s1_checks = _s1_certificate(normal, inverse, k, d)
    stats = {"s1_binary_patterns": s1_checks, "maximal_carrier_supports": 0,
             "binary_independent_supports": 0, "ring_eliminations": 0,
             "s2_primitive_kernel_vectors": 0}
    def reject(witness):
        _check_witness(u, a, inverse, k, d, witness)
        return {"passed": False, "surviving_witness": witness, **stats}
    if mask is not None:
        return reject({"s": 1, "a": [4*((mask >> j)&1) for j in range(m)],
                       "b": [4*((mask >> (m+j))&1) for j in range(m)]})
    parity = [sum((int(normal[i,j])&1) << i for i in range(m-k)) for j in range(2*m)]
    for support in combinations(range(m), d-1):
        stats["maximal_carrier_supports"] += 1
        positions = list(support) + [m+j for j in support]
        if _independent_binary_columns(parity[j] for j in positions):
            stats["binary_independent_supports"] += 1
            continue
        stats["ring_eliminations"] += 1
        decomposition = kernel_decomposition(normal[:,positions], 3, ncols=2*(d-1))
        mod4 = decomposition.at_modulus(2)
        if mod4.primitive_witness() is None:
            continue  # This also excludes primitive modulo-eight vectors.
        for word in mod4.vectors(primitive_only=True):
            stats["s2_primitive_kernel_vectors"] += 1
            if s2_cost(word) < d:
                return reject(_embed_witness(word, support, m, 2))
        witness8 = decomposition.primitive_witness()
        if witness8 is not None:
            return reject(_embed_witness(witness8, support, m, 3))
    return {"passed": True, "distance_guarantee": d, **stats,
            "method": "Exact Algorithm S2: binary s1; cost-aware mod4 s2; primitive mod8 s3",
            "arithmetic": "exact integers modulo 2, 4 and 8"}


def self_test() -> dict:
    rng = random.Random(2026100917)
    kernel_checks = 0
    for s in (1,2,3):
        modulus = 1 << s
        for nr in range(4):
            for nc in range(4):
                for _ in range(6):
                    matrix = [[rng.randrange(modulus) for _ in range(nc)] for _ in range(nr)]
                    decomposition = kernel_decomposition(matrix,s,ncols=nc)
                    expected = {word for word in product(range(modulus),repeat=nc)
                                if all(sum(x*y for x,y in zip(row,word)) % modulus == 0
                                       for row in matrix)}
                    actual = {tuple(word) for word in decomposition.vectors()}
                    assert actual == expected, (matrix,s,actual,expected)
                    assert (decomposition.primitive_witness() is not None) == any(
                        any(x&1 for x in word) for word in expected)
                    if s > 1:
                        reduced = decomposition.at_modulus(s-1)
                        expected_low = {word for word in product(range(modulus//2),repeat=nc)
                                        if all(sum(x*y for x,y in zip(row,word)) % (modulus//2) == 0
                                               for row in matrix)}
                        assert {tuple(word) for word in reduced.vectors()} == expected_low
                    kernel_checks += 1
    literal_checks = 0
    for m in (2,3,4,5):
        for k in range(1,m):
            for d in range(1,min(m,3)+1):
                for _ in range(4):
                    u,a = _literal.sample_candidate(m,rng)
                    fast = verify_current_algorithm(u,a,k,d)
                    slow = _literal.verify_distance_certificate(u,a,k,d)
                    assert fast['passed'] == slow['passed'], (m,k,d,fast,slow)
                    literal_checks += 1
    # Isolate the higher strata so early s1 rejections cannot hide a bug in
    # either the cost filter or the distinction between modulo four and eight.
    def higher_flags(matrix, m, d):
        fast = {2: False, 3: False}
        array = np.asarray(matrix, dtype=np.int64)
        for support in combinations(range(m), min(m,d-1)):
            positions = list(support)+[m+j for j in support]
            decomposition = kernel_decomposition(array[:,positions],3)
            fast[2] |= any(s2_cost(word) < d for word in
                           decomposition.at_modulus(2).vectors(primitive_only=True))
            fast[3] |= decomposition.primitive_witness() is not None
        slow = {2: False, 3: False}
        for s, a, b in _literal.forbidden_terms(m,d):
            if s >= 2 and all(sum(x*y for x,y in zip(row,a+b)) % 8 == 0 for row in matrix):
                slow[s] = True
        assert fast == slow, (matrix,m,d,fast,slow)
        return fast

    crafted = [([[6,1],[4,0]], {2:False,3:False}),
               ([[6,1],[0,0]], {2:False,3:True}),
               ([[7,1],[4,0]], {2:True,3:False}),
               ([[7,1],[0,0]], {2:True,3:True})]
    for matrix, expected in crafted:
        assert higher_flags(matrix,1,2) == expected
    higher_checks = len(crafted)
    for m in (1,2,3):
        for nr in (1,2,3,4):
            for d in range(2,min(m+1,3)+1):
                for _ in range(3):
                    matrix = [[rng.randrange(8) for _ in range(2*m)] for _ in range(nr)]
                    higher_flags(matrix,m,d)
                    higher_checks += 1
    return {"passed": True, "exact_kernel_set_comparisons": kernel_checks,
            "literal_forbidden_terms_comparisons": literal_checks,
            "isolated_s2_s3_literal_comparisons": higher_checks,
            "crafted_cost_and_modulus_cases": len(crafted),
            "tested_moduli": [2,4,8], "tested_distances": [1,2,3]}
