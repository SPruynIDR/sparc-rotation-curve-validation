# data/

This directory is intentionally empty in the repository. It is where
`python src/download_data.py` places the two SPARC data tables this
analysis uses:

| File | Source | Contents |
|---|---|---|
| `SPARC_Lelli2016c.mrt` | https://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt | Galaxy sample table (Table 1): `L_[3.6]`, `M_HI`, `R_HI`, Hubble type `T`, quality flag `Q` |
| `MassModels_Lelli2016c.mrt` | https://astroweb.cwru.edu/SPARC/MassModels_Lelli2016c.mrt | Per-point mass models (Table 2): `r`, `V_obs`, `V_gas`, `V_disk`, `V_bul` |

**Why the data isn't committed to this repo:** SPARC is maintained and
published by Lelli, McGaugh & Schombert (2016, AJ, 152, 157). Downloading
it fresh from the canonical source, rather than bundling a redistributed
copy, guarantees every user gets the actual published files with correct
attribution, and the repository stays small.

**Citation, if you use this data:** Lelli, F., McGaugh, S. S., & Schombert,
J. M. 2016, AJ, 152, 157. Mirrored on VizieR/CDS at DOI
10.26093/cds/vizier.51520157 if astroweb.cwru.edu is temporarily
unreachable.

## Byte-by-byte column resolution

Both files are standard CDS/VizieR machine-readable tables. `src/parse_mrt.py`
reads them with astropy's built-in CDS reader by default, with a manual
byte-offset fallback. On first run, `src/sparc_validation.py` prints every
column name it found in each file — check that output against
`src/sparc_validation.py`'s `TABLE1_CANDIDATES` / `TABLE2_CANDIDATES`
dictionaries if a column fails to resolve; edit the candidate list there,
not the analysis logic, if a label doesn't match.
