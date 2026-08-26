"""
test_reproducibility.py — automated check that the pipeline still
reproduces the paper's headline numbers, to the tolerances stated in
results/VERIFIED_RESULTS.txt.

This requires the real SPARC data (run `python src/download_data.py`
first, or let the GitHub Actions workflow do it — see
.github/workflows/reproduce.yml). It is deliberately NOT mocked: the
point of this test is to catch drift in the actual public data pipeline,
not just in the local model code.

Run:
    pytest tests/test_reproducibility.py -v
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_PRESENT = (DATA_DIR / "SPARC_Lelli2016c.mrt").exists() and (
    DATA_DIR / "MassModels_Lelli2016c.mrt"
).exists()

pytestmark = pytest.mark.skipif(
    not DATA_PRESENT,
    reason="SPARC data not downloaded — run `python src/download_data.py` first",
)


@pytest.fixture(scope="module")
def analysis():
    from sparc_validation import load_data, compute_predictions, rms, r_squared

    rows, galaxy_table, excluded = load_data()
    Vobs, V_pred, V_bar = compute_predictions(rows)
    return dict(
        rows=rows,
        galaxy_table=galaxy_table,
        excluded=excluded,
        Vobs=Vobs,
        V_pred=V_pred,
        V_bar=V_bar,
        rms=rms,
        r_squared=r_squared,
    )


def test_galaxy_count(analysis):
    assert len(analysis["galaxy_table"]) == 171


def test_excluded_galaxies(analysis):
    expected = {"D512-2", "D564-8", "D631-7", "NGC5907"}
    assert set(analysis["excluded"]) == expected


def test_point_count(analysis):
    assert len(analysis["rows"]) == 3346


def test_headline_rms(analysis):
    this_work_rms = analysis["rms"](analysis["Vobs"], analysis["V_pred"])
    assert this_work_rms == pytest.approx(32.0117, abs=0.01)


def test_baryons_only_rms(analysis):
    baryons_rms = analysis["rms"](analysis["Vobs"], analysis["V_bar"])
    assert baryons_rms == pytest.approx(43.9003, abs=0.05)


def test_r_squared(analysis):
    r2 = analysis["r_squared"](analysis["Vobs"], analysis["V_pred"])
    assert r2 == pytest.approx(0.8666, abs=0.005)
