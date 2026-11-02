"""Exact cyclic-Z8 utilities shared by the finite verifier and generator.

The literal forbidden-term enumerator independently checks the accelerated
verifier on small inputs. search() and exhaustive() provide finite reference
searches; their costs can be exponential. Use the repository-root construct.py command
for the stored 48- and 90-qubit examples. There is no command-line entry point.
"""
from fractions import Fraction as F
from itertools import combinations, product
from math import comb
import random

ONE_QUBIT_EVEN = {(0, 2), (0, 4), (0, 6), (4, 0)} | {
    (a, b) for a in (2, 6) for b in (0, 2, 6)
}

def finite_bound(m, k, d):
    """Exact sufficient failure-probability bound from the manuscript."""
    if not (1 <= k < m and 1 <= d <= m):
        raise ValueError("Require 1 <= K < M and 1 <= d <= M")
    r = m - k
    coeff = [1] + [0] * (d - 1)
    for _ in range(m):
        coeff = [coeff[w] + (10 * coeff[w - 1] if w else 0)
                 + (5 * coeff[w - 2] if w >= 2 else 0) for w in range(d)]
    return (F(sum(comb(2 * m, w) for w in range(1, d)), 2**m)
            + F(sum(comb(m, w) for w in range(1, d)), 2**r)
            + F(sum(coeff[1:]), 2**(2 * r))
            + F(sum(63**w * comb(m, w) for w in range(1, d)), 2**(3 * r)))


def inverse_mod8(matrix):
    m = len(matrix)
    aug = [list(row) + [int(i == j) for j in range(m)]
           for i, row in enumerate(matrix)]
    for j in range(m):
        pivot = next((i for i in range(j, m) if aug[i][j] & 1), None)
        if pivot is None:
            return None
        aug[j], aug[pivot] = aug[pivot], aug[j]
        inv = pow(aug[j][j], -1, 8)
        aug[j] = [(v * inv) % 8 for v in aug[j]]
        for i in range(m):
            if i != j:
                scale = aug[i][j]
                aug[i] = [(x - scale * y) % 8 for x, y in zip(aug[i], aug[j])]
    return [row[m:] for row in aug]


def row_product(row, matrix):
    return [sum(row[j] * matrix[j][col] for j in range(len(row))) % 8
            for col in range(len(matrix[0]))]


def forbidden_terms(m, d):
    """Yield each exact-stratum pair in the three sufficient bad-term lists."""
    alphabets = {
        1: [(4, 0, 1), (0, 4, 1), (4, 4, 2)],
        2: [(a, b, 1 if (a, b) in ONE_QUBIT_EVEN else 2)
            for a in (0, 2, 4, 6) for b in (0, 2, 4, 6) if a or b],
        3: [(a, b, 1) for a in range(8) for b in range(8) if a or b],
    }
    for s, alphabet in alphabets.items():
        valuation_bit = 1 << (3 - s)
        for weight in range(1, min(m, d - 1) + 1):
            for positions in combinations(range(m), weight):
                for local in product(alphabet, repeat=weight):
                    if sum(v[2] for v in local) >= d:
                        continue
                    if not any((a | b) & valuation_bit for a, b, _ in local):
                        continue
                    a, b = [0] * m, [0] * m
                    for j, (aj, bj, _) in zip(positions, local):
                        a[j], b[j] = aj, bj
                    yield s, a, b


def verify_distance_certificate(u, a_matrix, k, d):
    """Sound sufficient distance test; a rejected candidate may still be good."""
    m = len(u)
    if not (1 <= k < m and 1 <= d <= m):
        raise ValueError("Invalid code parameters")
    for matrix in (u, a_matrix):
        if len(matrix) != m or any(len(row) != m for row in matrix):
            raise ValueError("Matrices must be square with the same dimension")
        if any(type(v) is not int or not 0 <= v < 8 for row in matrix for v in row):
            raise ValueError("Matrix entries must be integer representatives 0..7")
    if any(a_matrix[i][j] != a_matrix[j][i] for i in range(m) for j in range(m)):
        raise ValueError("A must be symmetric")
    inv = inverse_mod8(u)
    if inv is None:
        return {"passed": False, "reason": "U is singular modulo two"}
    r = m - k
    checked = 0
    for s, a, b in forbidden_terms(m, d):
        checked += 1
        if s == 1 and any(row_product(a, inv)[r:]):
            # The logical shift is 4 in some coordinate, hence leaves {0,1}^K.
            continue
        residual = [(bj - aj) % 8 for bj, aj in zip(b, row_product(a, a_matrix))]
        if all(sum(x * y for x, y in zip(row, residual)) % 8 == 0 for row in u[:r]):
            return {"passed": False, "checked_terms": checked,
                    "surviving_witness": {"s": s, "a": a, "b": b}}
    return {"passed": True, "checked_terms": checked, "distance_guarantee": d}


def make_instance(u, a_matrix, k, d, certificate):
    m, inv = len(u), inverse_mod8(u)
    if not certificate.get("passed") or inv is None:
        raise ValueError("A passing distance certificate is required")
    r = m - k
    f = [[(inv[j][r + ell] << bit) % 8 for j in range(m) for bit in range(3)]
         for ell in range(k)]
    # Exact selector identity U*z_ell^T = e_(r+ell)^T.
    for ell in range(k):
        z = [inv[j][r + ell] for j in range(m)]
        assert [sum(x*y for x, y in zip(row, z)) % 8 for row in u] == [
            int(j == r + ell) for j in range(m)]
    return {"status": "distance_and_addressability_certified", "M": m,
            "N": 3*m, "K": k, "distance_guarantee": d,
            "rate": str(F(k, 3*m)), "relative_distance_guarantee": str(F(d, 3*m)),
            "U_mod8": u, "A_mod8": a_matrix, "F_mod8_lsb_first": f,
            "encoding": "E|x>=8^(-r/2) sum_v exp(pi*i*y*A*y^T/8)|dig(y)>; y=vG+xL mod 8",
            "r": r, "quadratic_phase_modulus": 16,
            "distance_certificate": certificate}


def sample_candidate(m, rng):
    while True:
        u = [[rng.randrange(8) for _ in range(m)] for _ in range(m)]
        if inverse_mod8(u) is not None:
            break
    a = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i, m):
            a[i][j] = a[j][i] = rng.randrange(8)
    return u, a


def search(m, k, d, max_trials=100, seed=None):
    # The probability theorem assumes independent uniform randomness.
    # An optional PRNG seed is for reproducible examples, not a derandomization.
    rng = random.SystemRandom() if seed is None else random.Random(seed)
    for trial in range(1, max_trials + 1):
        u, a = sample_candidate(m, rng)
        certificate = verify_distance_certificate(u, a, k, d)
        if certificate["passed"]:
            result = make_instance(u, a, k, d, certificate)
            result["accepted_trial"] = trial
            result["sampling"] = "system randomness" if seed is None else f"PRNG seed {seed}"
            return result
    raise RuntimeError("Trial budget exhausted; no certified code was returned")


def exhaustive(m, k, d):
    """Lexicographic finite search, guaranteed to succeed when finite_bound<1."""
    for entries in product(range(8), repeat=m*m):
        u = [list(entries[i*m:(i+1)*m]) for i in range(m)]
        if inverse_mod8(u) is None:
            continue
        positions = [(i, j) for i in range(m) for j in range(i, m)]
        for upper in product(range(8), repeat=len(positions)):
            a = [[0] * m for _ in range(m)]
            for (i, j), v in zip(positions, upper):
                a[i][j] = a[j][i] = v
            certificate = verify_distance_certificate(u, a, k, d)
            if certificate["passed"]:
                result = make_instance(u, a, k, d, certificate)
                result["selection"] = "lexicographically first accepted pair"
                return result
    raise RuntimeError("No candidate passed this sufficient certificate")


def physical_powers(instance, logical):
    if len(logical) != instance["K"] or any(type(v) is not int for v in logical):
        raise ValueError("Supply one integer exponent per logical qubit")
    return row_product(logical, instance["F_mod8_lsb_first"])
