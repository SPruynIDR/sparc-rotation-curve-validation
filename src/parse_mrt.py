"""
parse_mrt.py — read a CDS/VizieR machine-readable table (.mrt) file.

SPARC's two tables (SPARC_Lelli2016c.mrt, MassModels_Lelli2016c.mrt) are
published in the standard AAS/CDS machine-readable format: a
"Byte-by-byte Description" header block defines fixed-width columns, and
the data section follows. This is a well-defined, standard format, so we
use astropy's built-in CDS reader rather than hand-rolling a fragile
whitespace-split parser.

A hand-written fallback parser is included for environments without
astropy, or in case a given file trips the known astropy issue with
integer-typed columns in some SPARC tables
(https://github.com/astropy/astropy/issues/12972) — the fallback reads the
byte-by-byte description directly and slices each data line by byte
offset, which is immune to that bug.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Any

try:
    from astropy.io import ascii as astropy_ascii  # type: ignore
    from astropy.table import Table  # type: ignore
    _HAVE_ASTROPY = True
except ImportError:  # pragma: no cover
    _HAVE_ASTROPY = False

BYTE_RANGE_RE = re.compile(
    r"^\s*(\d+)-\s*(\d+)\s+\S+\s+\S+\s+(\S+)\s+"
)
SINGLE_BYTE_RE = re.compile(
    r"^\s*(\d+)\s+\S+\s+\S+\s+(\S+)\s+"
)


def _manual_parse(path: Path) -> dict[str, list[Any]]:
    """Fallback: parse the byte-by-byte header, then slice data lines by offset."""
    lines = path.read_text(errors="replace").splitlines()

    dash_idx = [i for i, ln in enumerate(lines) if set(ln.strip()) == {"-"} and len(ln.strip()) > 10]
    if len(dash_idx) < 3:
        raise ValueError(f"{path.name}: could not locate standard CDS header separators")

    header_start, header_end = dash_idx[1] + 1, dash_idx[2]
    columns: list[tuple[int, int, str]] = []
    for ln in lines[header_start:header_end]:
        m = BYTE_RANGE_RE.match(ln)
        if m:
            start, end, label = int(m.group(1)), int(m.group(2)), m.group(3)
            columns.append((start - 1, end, label))
            continue
        m = SINGLE_BYTE_RE.match(ln)
        if m:
            pos, label = int(m.group(1)), m.group(2)
            columns.append((pos - 1, pos, label))

    data_start = dash_idx[-1] + 1
    result: dict[str, list[Any]] = {label: [] for _, _, label in columns}
    for ln in lines[data_start:]:
        if not ln.strip():
            continue
        for start, end, label in columns:
            raw = ln[start:end].strip()
            result[label].append(raw)
    return result


def read_mrt(path: str | Path):
    """
    Read a CDS-format .mrt file and return an astropy Table if astropy is
    available, otherwise a dict of column-name -> list-of-strings (caller
    is responsible for type conversion in that case).
    """
    path = Path(path)
    if _HAVE_ASTROPY:
        try:
            return astropy_ascii.read(str(path), format="cds")
        except Exception:
            pass
    return _manual_parse(path)


def column(table, name: str, dtype=float):
    """Get a column as a numpy array regardless of whether `table` is an
    astropy Table or the manual-parser dict fallback."""
    import numpy as np

    if _HAVE_ASTROPY and isinstance(table, Table):
        return np.asarray(table[name], dtype=dtype)
    return np.array([dtype(v.split()[0]) if v and str(v).strip() not in ("", "...") and len(str(v).split()) > 0 else np.nan for v in table[name]])
    
