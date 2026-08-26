# Zero Per-Galaxy Free Parameters: A Global Boundary-Scaling Relation for 171 SPARC Rotation Curves

[

![Reproducibility check](https://github.com/SPruynIDR/sparc-rotation-curve-validation/actions/workflows/reproduce.yml/badge.svg)

](https://github.com/SPruynIDR/sparc-rotation-curve-validation/actions/workflows/reproduce.yml)
[

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

](LICENSE)
[

![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22068114.svg)

](https://doi.org/10.5281/zenodo.22068114)

Reproduction code and data pipeline for:

> Pruyn, S. M. (2026). *Zero Per-Galaxy Free Parameters: A Global
> Boundary-Scaling Relation for 171 SPARC Rotation Curves.* Zenodo.
> https://doi.org/10.5281/zenodo.22068114

This repository is the code companion to the paper above, submitted to
arXiv (astro-ph.GA) and MNRAS. It is self-contained: cloning it and
running three commands reproduces every headline number, table, and
figure in the paper from raw, public data — no bundled or pre-processed
data files, no hidden manual steps.

**If any part of this pipeline fails to reproduce the numbers in
`results/VERIFIED_RESULTS.txt`, please open an issue.** That is
exactly the kind of report this repository exists to catch.

---

## 1. What this tests

A single functional form is fit **once**, globally, across every point
in the SPARC sample (Lelli, McGaugh & Schombert 2016), and then compared
— with the constants **frozen**, never refit — against galaxy rotation
curves it was not individually tuned to:
