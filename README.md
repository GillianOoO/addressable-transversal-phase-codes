# Addressable transversal phase codes

Code and construction data accompanying **Near-Optimal Rate–Distance Bounds for Cyclic Encodings with Addressable Transversal Phase Gates**, by **Bujiao Wu, Tongyin Lin, and Bin Cheng**.

This repository contains the random quadratic-phase construction at **h = 3**, the defining matrices of the two finite examples, and the native-operation resource summary for the Q48 encoder. The implementation works over Z/8Z and supports fully addressable transversal T powers. The manuscript establishes the construction for general fixed h; the software deposited here implements h = 3.

## Finite examples

| Code | Physical qubits N | Logical qubits K | Distance D in the manuscript | h | Ring registers M | Uniform ring inputs r |
|---|---:|---:|---:|---:|---:|---:|
| Q48 | 48 | 4 | 5 | 3 | 16 | 12 |
| Q90 | 90 | 8 | 7 | 3 | 30 | 22 |

Each `code.json` contains `B_mod8`, `A_mod8`, these basic parameters, and the generation settings. K counts logical qubits; the code-space dimension is 2^K. The stored D values are the exact distances reported in the manuscript. The construction acceptance test checks the corresponding lower bounds; the matching logical-operator argument is given in the manuscript.

## Files

```text
construct.py                         Command-line entry point
src/core/ring8.py                     Exact ring arithmetic and literal search
src/exponential/construction.py      Seeded generation and instance commands
src/verification/rejection.py        Accelerated exact rejection predicate
data/finite/q48/code.json             Q48 defining matrices and parameters
data/finite/q90/code.json             Q90 defining matrices and parameters
data/source_provenance.json          Source hashes and packaging adaptations
results/validation.json              Commands and results from release checks
MANIFEST.json                        SHA-256 inventory of release files
CITATION.cff / citation.bib           Citation metadata
```

## Installation

Use Python 3.10 or later and NumPy. From the repository root:

```sh
python -m pip install -r requirements.txt
```

The preparation checks used CPython 3.14.7 and NumPy 2.5.2 on Windows. `requirements-tested.txt` records the tested NumPy version. The commands below also work when `construct.py` is invoked by its absolute path from another directory.

## Reproduce the two constructions

```sh
python construct.py reproduce --seed 202610011 --max-trials 10000 --output-dir generated/reproduced
```

One `random.Random` stream is used for Q48 first and Q90 second. It accepts candidate **2** for Q48 and candidate **1** for Q90. The command compares the resulting B and A matrices with the deposited matrices and reports `matches_deposited_matrices: true` for both examples. To reproduce Q90, retain this shared-stream order.

The generator samples B until it is invertible modulo eight, samples the upper triangle of a symmetric A independently, and accepts a pair after the exact rejection test passes. Invertibility is equivalent to invertibility modulo two. The fixed seed reproduces the finite examples; the probability analysis in the manuscript assumes independent uniform samples. A finite trial cap can be exhausted for other targets.

Outputs go to the specified directory. Existing outputs require `--overwrite`; writes into the deposited `data/` directory are refused.

## Check saved construction inputs

```sh
python construct.py verify
python construct.py verify --input-dir generated/reproduced
python construct.py self-test
```

`verify` applies the complete construction rejection predicate to both saved examples and checks the inverse-column logical phase identities. It treats the order-two, order-four and order-eight lists exactly. The accelerated implementation uses binary screening and exact modular elimination; NumPy forms integer matrices, and all acceptance decisions use exact arithmetic. `self-test` compares the accelerated routines with literal enumeration on small cases.

The command also reports the exact finite ensemble failure bound. This sufficient bound is greater than one for Q48 and less than one for Q90. Both particular candidates pass the exact acceptance test: a sufficient ensemble inequality is not a necessary condition for an individual code to pass.

On the preparation machine, regenerating the two examples and verifying each pair took seconds; see `results/validation.json` for measured times. Costs grow rapidly with the requested target. The accelerated finite verifier uses a meet-in-the-middle table whose memory can grow exponentially with the target distance. The literal verifier streams the forbidden terms and supplies the reference procedure corresponding to the manuscript's polynomial-working-space search.

## Other finite targets

Bounded random search at h = 3:

```sh
python construct.py search --M 6 --K 1 --distance 2 --seed 20261102 --max-trials 100 --output generated/custom.json
```

Omit `--seed` to use system randomness. Custom outputs contain a certified `distance_lower_bound`, without claiming an exact minimum distance.

The existing deterministic reference implementation enumerates admissible matrices lexicographically and uses the literal rejection predicate. It is intended for tiny targets:

```sh
python construct.py exhaustive --M 2 --K 1 --distance 1 --output generated/tiny.json
```

## Encoding and data conventions

Let G be the first r rows of B and L the next K rows. For a binary logical label x, the encoding is

```text
E|x> = 8^(-r/2) sum_{v in (Z/8Z)^r}
       exp(pi*i*y*A*y^T/8) |dig(y)>,   y = vG + xL mod 8.
```

- Ring coordinates and matrices use representatives 0,...,7. The symmetric integer lift of A is used in the quadratic expression, whose exponent is evaluated modulo 16.
- Each register contains three physical qubits in least-to-most-significant digit order.
- `B_mod8` is the manuscript matrix B. Internal arithmetic variables named `u` and legacy API keys named `U_mod8` refer to the same matrix.
- Code indices are zero-based; displayed indices in the manuscript are one-based. Code register j contains physical qubit indices 3j, 3j+1 and 3j+2.
- Column r+ell of B inverse gives z_ell. The physical addressing coefficients are `F[ell,3*j+b] = 2^b*z_ell[j] mod 8`. The routines compute these when needed; the defining data files store B and A.

## Q48 native encoding resources

An ancilla-free encoding circuit for Q48 uses:

| Native operation | Number of qubits | Count |
|---|---:|---:|
| Controlled-phase gate | 2 | 466 |
| SWAP gate | 2 | 24 |
| Multiplication `x -> u*x mod 8`, with odd u | 3 | 12 |
| Weighted addition `(x,y) -> (x,y+c*x mod 8)` | 6 | 202 |

Scheduling these operations on disjoint qubits in parallel gives a **multi-qubit gate depth of 228**, assuming all-to-all connectivity and unit duration for every listed operation. This depth counts the listed multi-qubit operations. The resource summary refers to `data/finite/q48/code.json` and uses the native arithmetic operations as the gate library.

## Result-to-command map

| Material | Input | Command |
|---|---|---|
| Q48 and Q90 construction | Seed 202610011; Q48 then Q90 | `python construct.py reproduce` |
| Acceptance and addressing checks | `data/finite/{q48,q90}/code.json` | `python construct.py verify` |
| Independent small-case comparisons | Deterministic cases in the verifier | `python construct.py self-test` |
| Other h=3 targets | M, K and target distance | `python construct.py search ...` |
| Deterministic reference search | A tiny h=3 target | `python construct.py exhaustive ...` |
| Q48 native resource summary | Q48 and the manuscript's native gate convention | Table above |

Refresh the file inventory after intentional changes with `python construct.py manifest`. Generated outputs and the manifest itself are excluded from its hashes.

## Citation and license

Please cite the accompanying manuscript and this repository. `CITATION.cff` contains structured metadata, and `citation.bib` supplies a BibTeX entry. Repository: https://github.com/GillianOoO/addressable-transversal-phase-codes. This initial repository is private; access requires owner authorization.

The code and accompanying data are provided under the MIT License; see `LICENSE`.
