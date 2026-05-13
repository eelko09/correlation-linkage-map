# Correlation Linkage Map

Build peptide-peptide correlation outputs and peptide linkage maps from a Z-score table.

## What this script does

`correlation_linkage_map.py`:
- Reads metadata + one Z-score input file
- Log-normalizes Z scores with `log2(z + 8) - 3`
- Filters peptides by minimum reactive sample count
- Computes cross-species pairwise Pearson correlations
- Writes:
  - `{output_prefix}_correlations_pass_thresh.tsv`
  - `{output_prefix}_linkage_map.tsv`
- Optional: streaming/chunked mode for lower memory usage
- Optional: scatterplot PNG output for peptide comparisons

## Input requirements

### Metadata TSV (`--metadata`)
Required columns:
- `CodeName`
- `SpeciesID`
- `Species`
- `Peptide`

### Z-score TSV (`--zscore-file`)
- First column should be peptide identifier (typically `Sequence name`)
- Remaining columns are sample values
- Non-numeric values are coerced to NaN

## Outputs

### 1) Correlation output
File: `{output_prefix}_correlations_pass_thresh.tsv`

Contains pair-level rows including:
- `CodeName1`, `CodeName2`
- `Pearson_Corr.`
- `OverlapSamples`
- metadata columns (`Species1`, `Species2`, `SpeciesID1`, `SpeciesID2`, `Peptide1`, `Peptide2`)
- optional `7mer_Ovlp` (if `--compute-7mer-overlap`)

Only cross-species pairs are retained (`Species1 != Species2`).

### 2) Linkage map output
File: `{output_prefix}_linkage_map.tsv`

Headers:
- `Peptide Name`
- `Species`

Species values are emitted as comma-separated `taxid:1` entries (unquoted TSV field).

## Installation / dependencies

Python 3.9+ recommended.

Install dependencies:

```bash
pip install pandas numpy matplotlib
```

## Basic usage

```bash
python correlation_linkage_map.py \
  --metadata /path/to/metadata.tsv \
  --zscore-file /path/to/zscores.tsv \
  --pearson-threshold 0.5 \
  --output-prefix PM1
```

## Full example (all major options)

```bash
python correlation_linkage_map.py \
  --metadata /path/to/metadata.tsv \
  --zscore-file /path/to/zscores.tsv \
  --drop-regex Sblk \
  --pearson-threshold 0.50 \
  --min-reactive-samples 130 \
  --z-threshold 10 \
  --min-shared-samples 40 \
  --block-size 512 \
  --stream-output \
  --stream-chunk-rows 200000 \
  --output-prefix PM1 \
  --compute-7mer-overlap \
  --scatterplot-mode above \
  --scatter-threshold 0.8 \
  --scatter-output-dir PM1_scatterplots \
  --scatter-max-plots 500
```

## Scatterplot options

- `--scatterplot-mode none|all|above|below`
  - `none`: disable plots (default)
  - `all`: plot all pairs in correlation output
  - `above`: plot pairs where `Pearson_Corr. >= --scatter-threshold`
  - `below`: plot pairs where `Pearson_Corr. <= --scatter-threshold`
- `--scatter-threshold FLOAT`: required for `above` / `below`
- `--scatter-output-dir DIR`: PNG output folder (default `{output_prefix}_scatterplots`)
- `--scatter-max-plots INT`: hard cap on number of plots
- `--scatter-peptides PEPTIDE1 PEPTIDE2`: generate a single pair plot

Example single pair plot:

```bash
python correlation_linkage_map.py \
  --metadata /path/to/metadata.tsv \
  --zscore-file /path/to/zscores.tsv \
  --pearson-threshold 0.5 \
  --output-prefix PM1 \
  --scatter-peptides IN2T_00025 IN2T_13620
```

## Performance notes

- Use `--stream-output` for very large runs to reduce peak memory.
- Streaming mode writes rows in processing order (not globally sorted by Pearson).
- In-memory mode sorts by `Pearson_Corr.` before writing.
- Tune `--block-size` and `--stream-chunk-rows` based on your laptop/server memory and CPU.

## Validation checks

```bash
python -m py_compile correlation_linkage_map.py
python correlation_linkage_map.py --help
```

## Common errors

- `--block-size must be >= 1`
- `--stream-chunk-rows must be >= 1`
- `--scatter-max-plots must be >= 1`
- Missing metadata columns (`CodeName`, `SpeciesID`, `Species`, `Peptide`)
- Missing `--scatter-threshold` when using `--scatterplot-mode above|below`
