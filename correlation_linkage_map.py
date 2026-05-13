#!/usr/bin/env python3
"""
Standalone peptide linkage builder from Z-score matrices.

Key points:
- Accepts metadata and a single Z-score file via CLI.
- Computes Pearson correlations only (Fisher's exact test removed).
- Uses blockwise vectorized math for pairwise-complete Pearson correlation.
- Filters to cross-species pairs and user-defined Pearson threshold.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for correlation and linkage-map generation."""
    parser = argparse.ArgumentParser(
        description="Build correlation linkage map from peptide Z-score matrix/matrices."
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Metadata TSV with at least CodeName, SpeciesID, Species, Peptide columns.",
    )
    parser.add_argument(
        "--zscore-file",
        required=True,
        help="Z-score TSV file (index should be peptide code names).",
    )
    parser.add_argument(
        "--pearson-threshold",
        type=float,
        required=True,
        help="Minimum Pearson correlation to keep a pair (e.g. 0.5).",
    )
    parser.add_argument(
        "--min-reactive-samples",
        type=int,
        default=130,
        help="Minimum number of reactive samples required for a peptide to be analyzed.",
    )
    parser.add_argument(
        "--reactive-z",
        type=float,
        default=10.0,
        help="Reactivity threshold in raw Z units before log normalization (default: 10).",
    )
    parser.add_argument(
        "--min-overlap-samples",
        type=int,
        default=40,
        help="Minimum shared non-NaN samples required to compute correlation.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="Peptides per block for vectorized pair computation.",
    )
    parser.add_argument(
        "--drop-regex",
        default="Sblk",
        help="Regex for sample columns to drop from each Z-score table.",
    )
    parser.add_argument(
        "--prefix",
        default="CorrelationLinkageMap",
        help="Prefix for output TSV files.",
    )
    parser.add_argument(
        "--compute-7mer-overlap",
        action="store_true",
        help="If set, compute 7-mer overlap for retained pairs (extra CPU/memory).",
    )
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help=(
            "Write high-correlation pairs in chunks to reduce peak memory usage "
            "(rows are not globally sorted in this mode)."
        ),
    )
    parser.add_argument(
        "--stream-chunk-rows",
        type=int,
        default=200000,
        help="Rows per chunk for streaming write/read stages.",
    )
    parser.add_argument(
        "--scatterplot-mode",
        choices=["none", "all", "above", "below"],
        default="none",
        help=(
            "Scatterplot output mode: none (default), all pairs in correlations output, "
            "or only pairs above/below --scatter-threshold."
        ),
    )
    parser.add_argument(
        "--scatter-threshold",
        type=float,
        default=None,
        help="Pearson threshold used with --scatterplot-mode above/below.",
    )
    parser.add_argument(
        "--scatter-output-dir",
        default=None,
        help="Directory for scatterplot PNGs. Default: {prefix}_scatterplots",
    )
    parser.add_argument(
        "--scatter-max-plots",
        type=int,
        default=2000,
        help="Maximum number of scatterplots to render (safety cap).",
    )
    parser.add_argument(
        "--scatter-peptides",
        nargs=2,
        metavar=("PEPTIDE1", "PEPTIDE2"),
        default=None,
        help="Render exactly one scatterplot for a specific peptide pair.",
    )
    return parser.parse_args()


def read_metadata(path: str) -> pd.DataFrame:
    """Read metadata and enforce required columns."""
    meta = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    needed = {"CodeName", "SpeciesID", "Species", "Peptide"}
    missing = needed.difference(meta.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    return meta


def read_one_zscore(path: str, drop_regex: str) -> pd.DataFrame:
    """Read a single Z-score TSV, set peptide index, normalize, and drop matched columns."""
    df = pd.read_csv(path, sep="\t")

    if "Sequence name" in df.columns:
        df = df.set_index("Sequence name")
    else:
        first = df.columns[0]
        df = df.set_index(first)

    df = df.apply(pd.to_numeric, errors="coerce")

    # Log normalization from notebook: log2(z + 8) - 3
    df = np.log2(df + 8.0) - 3.0

    if drop_regex:
        drop_cols = df.filter(regex=drop_regex).columns
        if len(drop_cols) > 0:
            df = df.drop(columns=drop_cols)

    return df


def kmer_set(seq: str, k: int) -> set:
    """Return unique k-mers for a peptide sequence."""
    if not isinstance(seq, str) or len(seq) < k:
        return set()
    return {seq[i : i + k] for i in range(0, len(seq) - k + 1)}


def normalize_species_id(s: str) -> str:
    """Normalize a species/taxon field to a base ID token used in linkage-map output."""
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
    s = s.split(";")[0].strip()
    # Convert values like 480:01:00 to 480, then append :1 when writing linkage map entries.
    return s.split(":")[0].strip()


def build_selected_matrix(
    zdf: pd.DataFrame,
    meta: pd.DataFrame,
    min_reactive_samples: int,
    reactive_z_raw: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str], Dict[str, str], Dict[str, str], List[str]]:
    """
    Build analysis-ready matrix and metadata maps.

    Returns:
    - X: samples x peptides float matrix after filtering on reactive sample count
    - peptides: peptide code names aligned to X columns
    - species_ids: SpeciesID values aligned to X columns
    - spid_map/sp_map/pepseq_map: metadata lookup dicts by CodeName
    """
    reactive_thresh_log = math.log2(reactive_z_raw + 8.0) - 3.0

    reactive_counts = (zdf > reactive_thresh_log).sum(axis=1)
    keep = reactive_counts[reactive_counts >= min_reactive_samples].index

    if len(keep) == 0:
        raise ValueError("No peptides pass --min-reactive-samples with current thresholds.")

    zdf_keep = zdf.loc[keep]

    # samples x peptides matrix
    X = zdf_keep.to_numpy(dtype=np.float64).T
    peptides = zdf_keep.index.to_numpy()

    spid_map = dict(zip(meta["CodeName"], meta["SpeciesID"]))
    sp_map = dict(zip(meta["CodeName"], meta["Species"]))
    pepseq_map = dict(zip(meta["CodeName"], meta["Peptide"]))

    species_ids = np.array([spid_map.get(p, "") for p in peptides], dtype=object)
    missing_spid = [p for p, s in zip(peptides, species_ids) if s == ""]
    if missing_spid:
        raise ValueError(
            f"{len(missing_spid)} peptides from Z-score data are missing SpeciesID in metadata. "
            "Example: " + ", ".join(missing_spid[:5])
        )

    return X, peptides, species_ids, spid_map, sp_map, pepseq_map, list(zdf.index)


def find_high_pairs(
    X: np.ndarray,
    peptides: np.ndarray,
    species_ids: np.ndarray,
    min_overlap: int,
    pearson_threshold: float,
    block_size: int,
) -> pd.DataFrame:
    """
    Compute cross-species Pearson hits in-memory.

    Uses blockwise matrix operations and pairwise-complete overlap handling with NaN masks.
    """
    p = X.shape[1]
    mask = ~np.isnan(X)
    X0 = np.where(mask, X, 0.0)
    out_chunks: List[pd.DataFrame] = []

    for i0 in range(0, p, block_size):
        i1 = min(i0 + block_size, p)
        Xi = X0[:, i0:i1]
        Mi = mask[:, i0:i1].astype(np.float64)
        spi = species_ids[i0:i1]
        uniq_spi = np.unique(spi)
        gi = {s: np.where(spi == s)[0] for s in uniq_spi}

        for j0 in range(i0, p, block_size):
            j1 = min(j0 + block_size, p)
            Xj = X0[:, j0:j1]
            Mj = mask[:, j0:j1].astype(np.float64)
            spj = species_ids[j0:j1]
            uniq_spj = np.unique(spj)
            gj = {s: np.where(spj == s)[0] for s in uniq_spj}

            # Split each block by species so we only compute cross-species pair math.
            for s1, idx_i in gi.items():
                for s2, idx_j in gj.items():
                    if s1 == s2:
                        continue
                    # In same block, keep one species-order direction to avoid duplicates.
                    if i0 == j0 and str(s1) >= str(s2):
                        continue

                    Xi_s = Xi[:, idx_i]
                    Mi_s = Mi[:, idx_i]
                    Xj_s = Xj[:, idx_j]
                    Mj_s = Mj[:, idx_j]

                    N = Mi_s.T @ Mj_s
                    valid = N >= min_overlap
                    if not np.any(valid):
                        continue

                    Sxy = Xi_s.T @ Xj_s
                    Sx = Xi_s.T @ Mj_s
                    Sy = Mi_s.T @ Xj_s
                    Sxx = (Xi_s * Xi_s).T @ Mj_s
                    Syy = Mi_s.T @ (Xj_s * Xj_s)

                    # Pearson r via sufficient statistics over pairwise-valid samples.
                    with np.errstate(divide="ignore", invalid="ignore"):
                        num = Sxy - (Sx * Sy) / N
                        den_left = Sxx - (Sx * Sx) / N
                        den_right = Syy - (Sy * Sy) / N
                        den = np.sqrt(den_left * den_right)
                        r = num / den

                    keep = valid & np.isfinite(r) & (r >= pearson_threshold)
                    if not np.any(keep):
                        continue

                    ii, jj = np.where(keep)
                    gi_idx = i0 + idx_i[ii]
                    gj_idx = j0 + idx_j[jj]
                    out_chunks.append(
                        pd.DataFrame(
                            {
                                "CodeName1": peptides[gi_idx],
                                "CodeName2": peptides[gj_idx],
                                "Pearson_Corr.": r[ii, jj].astype(np.float64),
                                "OverlapSamples": N[ii, jj].astype(np.int64),
                            }
                        )
                    )

    if not out_chunks:
        return pd.DataFrame(columns=["CodeName1", "CodeName2", "Pearson_Corr.", "OverlapSamples"])
    return pd.concat(out_chunks, ignore_index=True)


def stream_high_pairs_to_file(
    X: np.ndarray,
    peptides: np.ndarray,
    species_ids: np.ndarray,
    min_overlap: int,
    pearson_threshold: float,
    block_size: int,
    out_path: Path,
    chunk_rows: int,
) -> int:
    """
    Compute cross-species Pearson hits and stream them to disk.

    This avoids holding all passing rows in RAM. Output order is processing-order, not
    globally sorted by Pearson value.
    """
    mask = ~np.isnan(X)
    X0 = np.where(mask, X, 0.0)
    wrote_header = False
    kept_total = 0
    out_chunks: List[pd.DataFrame] = []
    buffered_rows = 0

    def flush() -> None:
        """Write buffered hit chunks to disk and reset buffer state."""
        nonlocal wrote_header, kept_total, out_chunks, buffered_rows
        if not out_chunks:
            return
        chunk_df = pd.concat(out_chunks, ignore_index=True)
        chunk_df.to_csv(
            out_path,
            sep="\t",
            index=False,
            mode="a",
            header=not wrote_header,
        )
        wrote_header = True
        kept_total += len(chunk_df)
        out_chunks = []
        buffered_rows = 0

    p = X.shape[1]
    for i0 in range(0, p, block_size):
        i1 = min(i0 + block_size, p)
        Xi = X0[:, i0:i1]
        Mi = mask[:, i0:i1].astype(np.float64)
        spi = species_ids[i0:i1]
        uniq_spi = np.unique(spi)
        gi = {s: np.where(spi == s)[0] for s in uniq_spi}

        for j0 in range(i0, p, block_size):
            j1 = min(j0 + block_size, p)
            Xj = X0[:, j0:j1]
            Mj = mask[:, j0:j1].astype(np.float64)
            spj = species_ids[j0:j1]
            uniq_spj = np.unique(spj)
            gj = {s: np.where(spj == s)[0] for s in uniq_spj}

            for s1, idx_i in gi.items():
                for s2, idx_j in gj.items():
                    if s1 == s2:
                        continue
                    if i0 == j0 and str(s1) >= str(s2):
                        continue

                    Xi_s = Xi[:, idx_i]
                    Mi_s = Mi[:, idx_i]
                    Xj_s = Xj[:, idx_j]
                    Mj_s = Mj[:, idx_j]

                    N = Mi_s.T @ Mj_s
                    valid = N >= min_overlap
                    if not np.any(valid):
                        continue

                    Sxy = Xi_s.T @ Xj_s
                    Sx = Xi_s.T @ Mj_s
                    Sy = Mi_s.T @ Xj_s
                    Sxx = (Xi_s * Xi_s).T @ Mj_s
                    Syy = Mi_s.T @ (Xj_s * Xj_s)

                    with np.errstate(divide="ignore", invalid="ignore"):
                        num = Sxy - (Sx * Sy) / N
                        den_left = Sxx - (Sx * Sx) / N
                        den_right = Syy - (Sy * Sy) / N
                        den = np.sqrt(den_left * den_right)
                        r = num / den

                    keep = valid & np.isfinite(r) & (r >= pearson_threshold)
                    if not np.any(keep):
                        continue

                    ii, jj = np.where(keep)
                    gi_idx = i0 + idx_i[ii]
                    gj_idx = j0 + idx_j[jj]
                    out_chunks.append(
                        pd.DataFrame(
                            {
                                "CodeName1": peptides[gi_idx],
                                "CodeName2": peptides[gj_idx],
                                "Pearson_Corr.": r[ii, jj].astype(np.float64),
                                "OverlapSamples": N[ii, jj].astype(np.int64),
                            }
                        )
                    )
                    buffered_rows += len(ii)
                    if buffered_rows >= chunk_rows:
                        flush()

    flush()
    return kept_total


def build_outputs(
    high_corr: pd.DataFrame,
    zdf_all: pd.DataFrame,
    spid_map: Dict[str, str],
    sp_map: Dict[str, str],
    pepseq_map: Dict[str, str],
    compute_7mer: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add metadata columns to hit table and build linkage-map species mapping."""
    high_corr = high_corr.copy()

    high_corr["Species1"] = high_corr["CodeName1"].map(sp_map).fillna("").str.split(";").str[0]
    high_corr["Species2"] = high_corr["CodeName2"].map(sp_map).fillna("").str.split(";").str[0]
    high_corr["SpeciesID1"] = high_corr["CodeName1"].map(spid_map).fillna("").str.split(";").str[0]
    high_corr["SpeciesID2"] = high_corr["CodeName2"].map(spid_map).fillna("").str.split(";").str[0]
    high_corr["Peptide1"] = high_corr["CodeName1"].map(pepseq_map).fillna("")
    high_corr["Peptide2"] = high_corr["CodeName2"].map(pepseq_map).fillna("")

    if compute_7mer:
        cache: Dict[str, set] = {}

        def get_kset(s: str) -> set:
            if s not in cache:
                cache[s] = kmer_set(s, 7)
            return cache[s]

        high_corr["7mer_Ovlp"] = [
            len(get_kset(a).intersection(get_kset(b)))
            for a, b in zip(high_corr["Peptide1"], high_corr["Peptide2"])
        ]

    corr_peps = set(high_corr["CodeName1"].tolist() + high_corr["CodeName2"].tolist())
    all_peps = set(zdf_all.index.tolist())
    no_corr = all_peps.difference(corr_peps)

    outD: Dict[str, str] = {c: "" for c in corr_peps}

    for row in high_corr.itertuples(index=False):
        pairs = [
            (row.CodeName1, row.SpeciesID1, row.SpeciesID2),
            (row.CodeName2, row.SpeciesID1, row.SpeciesID2),
        ]

        for codename, s1, s2 in pairs:
            s1 = normalize_species_id(s1)
            s2 = normalize_species_id(s2)
            cur = outD.get(codename, "")
            if not cur:
                if s1 == s2:
                    outD[codename] = f"{s1}:1" if s1 else ""
                else:
                    vals = [f"{s}:1" for s in (s1, s2) if s]
                    outD[codename] = ",".join(vals)
                continue

            for s in (s1, s2):
                if s and f"{s}:" not in cur:
                    cur = f"{cur},{s}:1"
            outD[codename] = cur

    for pep in no_corr:
        sid = normalize_species_id(spid_map.get(pep, ""))
        if sid:
            outD[pep] = f"{sid}:1"

    linkage_df = pd.DataFrame.from_dict(outD, orient="index")
    return high_corr, linkage_df


def write_linkage_map(linkage_df: pd.DataFrame, out_path: Path) -> None:
    """Write linkage map with fixed output headers and unquoted TSV fields."""
    out_df = linkage_df.copy()
    out_df.columns = ["Species"]
    out_df.index.name = "Peptide Name"
    out_df.to_csv(
        out_path,
        sep="\t",
        header=True,
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )


def sanitize_filename_piece(value: str) -> str:
    """Return a filesystem-safe filename token."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))


def plot_pair_scatter(
    pep1: str,
    pep2: str,
    X: np.ndarray,
    pep_to_idx: Dict[str, int],
    out_dir: Path,
    pearson_val: float | None = None,
) -> bool:
    """
    Save one scatterplot for a peptide pair.

    Returns True if a plot was written, False when pair is unavailable or has <2 valid points.
    """
    if pep1 not in pep_to_idx or pep2 not in pep_to_idx:
        return False
    # Lazy import keeps non-scatterplot runs faster and avoids matplotlib startup warnings.
    import matplotlib.pyplot as plt

    i = pep_to_idx[pep1]
    j = pep_to_idx[pep2]
    x = X[:, i]
    y = X[:, j]
    mask = ~np.isnan(x) & ~np.isnan(y)
    if int(mask.sum()) < 2:
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(x[mask], y[mask], s=8, alpha=0.55, edgecolors="none")
    title = f"{pep1} vs {pep2}"
    if pearson_val is not None and np.isfinite(pearson_val):
        title += f" | r={pearson_val:.3f}"
    ax.set_title(title)
    ax.set_xlabel(pep1)
    ax.set_ylabel(pep2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_name = f"{sanitize_filename_piece(pep1)}__{sanitize_filename_piece(pep2)}.png"
    fig.savefig(out_dir / out_name, dpi=140)
    plt.close(fig)
    return True


def generate_scatterplots(
    args: argparse.Namespace,
    corr_out: Path,
    X: np.ndarray,
    peptides: np.ndarray,
) -> int:
    """Generate scatterplots based on CLI mode and correlation output file."""
    out_dir = Path(args.scatter_output_dir or f"{args.prefix}_scatterplots")
    pep_to_idx = {p: idx for idx, p in enumerate(peptides.tolist())}

    if args.scatter_peptides:
        p1, p2 = args.scatter_peptides
        wrote = plot_pair_scatter(p1, p2, X, pep_to_idx, out_dir, pearson_val=None)
        return 1 if wrote else 0

    mode = args.scatterplot_mode
    if mode == "none":
        return 0
    if mode in {"above", "below"} and args.scatter_threshold is None:
        raise ValueError("--scatter-threshold is required with --scatterplot-mode above/below.")

    if not corr_out.exists():
        return 0

    made = 0
    for chunk in pd.read_csv(corr_out, sep="\t", chunksize=max(10000, args.stream_chunk_rows)):
        if mode == "above":
            chunk = chunk[chunk["Pearson_Corr."] >= args.scatter_threshold]
        elif mode == "below":
            chunk = chunk[chunk["Pearson_Corr."] <= args.scatter_threshold]

        if chunk.empty:
            continue

        for _, row in chunk.iterrows():
            ok = plot_pair_scatter(
                row["CodeName1"],
                row["CodeName2"],
                X,
                pep_to_idx,
                out_dir,
                pearson_val=float(row.get("Pearson_Corr.", np.nan)),
            )
            if ok:
                made += 1
                if made >= args.scatter_max_plots:
                    return made
    return made


def main() -> None:
    """Run end-to-end pipeline in streaming or in-memory mode."""
    # Pipeline overview:
    # 1) Read metadata + z-score table and normalize values.
    # 2) Filter peptides by minimum reactive samples.
    # 3) Compute cross-species Pearson correlations above threshold.
    # 4) Build two outputs:
    #    - correlations_pass_thresh.tsv (pair-level table)
    #    - linkage_map.tsv (peptide -> linked species mapping)
    start_time = time.perf_counter()
    args = parse_args()
    if args.block_size <= 0:
        raise ValueError("--block-size must be >= 1.")
    if args.stream_chunk_rows <= 0:
        raise ValueError("--stream-chunk-rows must be >= 1.")
    if args.scatter_max_plots <= 0:
        raise ValueError("--scatter-max-plots must be >= 1.")

    meta = read_metadata(args.metadata)
    zdf = read_one_zscore(args.zscore_file, drop_regex=args.drop_regex)

    X, peptides, species_ids, spid_map, sp_map, pepseq_map, _ = build_selected_matrix(
        zdf=zdf,
        meta=meta,
        min_reactive_samples=args.min_reactive_samples,
        reactive_z_raw=args.reactive_z,
    )

    corr_out = Path(f"{args.prefix}_correlations_pass_thresh.tsv")
    link_out = Path(f"{args.prefix}_linkage_map.tsv")
    # -------------------------------
    # Streaming mode: lower RAM usage
    # -------------------------------
    if args.stream_output:
        core_out = Path(f"{args.prefix}_correlations_core.tmp.tsv")
        if core_out.exists():
            core_out.unlink()
        if corr_out.exists():
            corr_out.unlink()

        prefilter_hits = stream_high_pairs_to_file(
            X=X,
            peptides=peptides,
            species_ids=species_ids,
            min_overlap=args.min_overlap_samples,
            pearson_threshold=args.pearson_threshold,
            block_size=args.block_size,
            out_path=core_out,
            chunk_rows=args.stream_chunk_rows,
        )

        if prefilter_hits == 0:
            pd.DataFrame(
                columns=["CodeName1", "CodeName2", "Pearson_Corr.", "OverlapSamples"]
            ).to_csv(corr_out, sep="\t", index=False)
            write_linkage_map(pd.DataFrame(columns=["Species"]), link_out)
            print("No pairs passed filters. Wrote empty outputs:")
            print(f"  {corr_out}")
            print(f"  {link_out}")
            elapsed = time.perf_counter() - start_time
            print(f"Elapsed time (s): {elapsed:.2f}")
            return

        corr_peps: set[str] = set()
        outD: Dict[str, str] = {}
        wrote_header = False
        written_rows = 0
        kmer_cache: Dict[str, set] = {}

        def get_kset(s: str) -> set:
            if s not in kmer_cache:
                kmer_cache[s] = kmer_set(s, 7)
            return kmer_cache[s]

        # Enrich and emit correlation output in chunks.
        for chunk in pd.read_csv(core_out, sep="\t", chunksize=args.stream_chunk_rows):
            chunk["Species1"] = chunk["CodeName1"].map(sp_map).fillna("").str.split(";").str[0]
            chunk["Species2"] = chunk["CodeName2"].map(sp_map).fillna("").str.split(";").str[0]
            chunk["SpeciesID1"] = chunk["CodeName1"].map(spid_map).fillna("").str.split(";").str[0]
            chunk["SpeciesID2"] = chunk["CodeName2"].map(spid_map).fillna("").str.split(";").str[0]
            chunk["Peptide1"] = chunk["CodeName1"].map(pepseq_map).fillna("")
            chunk["Peptide2"] = chunk["CodeName2"].map(pepseq_map).fillna("")
            chunk = chunk[chunk["Species1"] != chunk["Species2"]].copy()
            if chunk.empty:
                continue

            if args.compute_7mer_overlap:
                chunk["7mer_Ovlp"] = [
                    len(get_kset(a).intersection(get_kset(b)))
                    for a, b in zip(chunk["Peptide1"], chunk["Peptide2"])
                ]

            chunk.to_csv(corr_out, sep="\t", index=False, mode="a", header=not wrote_header)
            wrote_header = True
            written_rows += len(chunk)

            # Build linkage-map assignments incrementally while streaming.
            for row in chunk.itertuples(index=False):
                corr_peps.add(row.CodeName1)
                corr_peps.add(row.CodeName2)
                outD.setdefault(row.CodeName1, "")
                outD.setdefault(row.CodeName2, "")

                pairs = [
                    (row.CodeName1, row.SpeciesID1, row.SpeciesID2),
                    (row.CodeName2, row.SpeciesID1, row.SpeciesID2),
                ]
                for codename, s1, s2 in pairs:
                    s1 = normalize_species_id(s1)
                    s2 = normalize_species_id(s2)
                    cur = outD.get(codename, "")
                    if not cur:
                        if s1 == s2:
                            outD[codename] = f"{s1}:1" if s1 else ""
                        else:
                            vals = [f"{s}:1" for s in (s1, s2) if s]
                            outD[codename] = ",".join(vals)
                        continue
                    for s in (s1, s2):
                        if s and f"{s}:" not in cur:
                            cur = f"{cur},{s}:1"
                    outD[codename] = cur

        all_peps = set(zdf.index.tolist())
        for pep in all_peps.difference(corr_peps):
            sid = normalize_species_id(spid_map.get(pep, ""))
            if sid:
                outD[pep] = f"{sid}:1"

        if not wrote_header:
            pd.DataFrame(
                columns=[
                    "CodeName1",
                    "CodeName2",
                    "Pearson_Corr.",
                    "OverlapSamples",
                    "Species1",
                    "Species2",
                    "SpeciesID1",
                    "SpeciesID2",
                    "Peptide1",
                    "Peptide2",
                ]
            ).to_csv(corr_out, sep="\t", index=False)

        write_linkage_map(pd.DataFrame.from_dict(outD, orient="index"), link_out)
        core_out.unlink(missing_ok=True)
        print("Done.")
        print("Output mode: streaming")
        print(f"Peptides analyzed: {X.shape[1]}")
        print(f"Pairs passing Pearson/species-id prefilter: {prefilter_hits}")
        print(f"Pairs written after species-name filter: {written_rows}")
        print(f"High-correlation output: {corr_out}")
        print(f"Linkage-map output: {link_out}")
        n_scatter = generate_scatterplots(args=args, corr_out=corr_out, X=X, peptides=peptides)
        if n_scatter > 0:
            scatter_dir = Path(args.scatter_output_dir or f"{args.prefix}_scatterplots")
            print(f"Scatterplots written: {n_scatter} -> {scatter_dir}")
        elapsed = time.perf_counter() - start_time
        print(f"Elapsed time (s): {elapsed:.2f}")
        return

    # -----------------------------------
    # In-memory mode: faster when data is
    # manageable in RAM, sorted output.
    # -----------------------------------
    high_corr = find_high_pairs(
        X=X,
        peptides=peptides,
        species_ids=species_ids,
        min_overlap=args.min_overlap_samples,
        pearson_threshold=args.pearson_threshold,
        block_size=args.block_size,
    )

    if high_corr.empty:
        high_corr.to_csv(corr_out, sep="\t", index=False)
        write_linkage_map(pd.DataFrame(columns=["Species"]), link_out)
        print("No pairs passed filters. Wrote empty outputs:")
        print(f"  {corr_out}")
        print(f"  {link_out}")
        elapsed = time.perf_counter() - start_time
        print(f"Elapsed time (s): {elapsed:.2f}")
        return

    # Apply species-name filter before enrichment/linkage build to avoid redundant work.
    high_corr = high_corr.copy()
    high_corr["Species1"] = high_corr["CodeName1"].map(sp_map).fillna("").str.split(";").str[0]
    high_corr["Species2"] = high_corr["CodeName2"].map(sp_map).fillna("").str.split(";").str[0]
    high_corr = high_corr[high_corr["Species1"] != high_corr["Species2"]].copy()
    if high_corr.empty:
        high_corr.to_csv(corr_out, sep="\t", index=False)
        write_linkage_map(pd.DataFrame(columns=["Species"]), link_out)
        print("No cross-species pairs remained after species-name filtering. Wrote empty outputs:")
        print(f"  {corr_out}")
        print(f"  {link_out}")
        elapsed = time.perf_counter() - start_time
        print(f"Elapsed time (s): {elapsed:.2f}")
        return
    high_corr, linkage_df = build_outputs(
        high_corr=high_corr,
        zdf_all=zdf,
        spid_map=spid_map,
        sp_map=sp_map,
        pepseq_map=pepseq_map,
        compute_7mer=args.compute_7mer_overlap,
    )[1]

    high_corr.sort_values("Pearson_Corr.", ascending=False).to_csv(corr_out, sep="\t", index=False)
    write_linkage_map(linkage_df, link_out)

    print("Done.")
    print("Output mode: in-memory")
    print(f"Peptides analyzed: {X.shape[1]}")
    print(f"Pairs kept: {len(high_corr)}")
    print(f"High-correlation output: {corr_out}")
    print(f"Linkage-map output: {link_out}")
    n_scatter = generate_scatterplots(args=args, corr_out=corr_out, X=X, peptides=peptides)
    if n_scatter > 0:
        scatter_dir = Path(args.scatter_output_dir or f"{args.prefix}_scatterplots")
        print(f"Scatterplots written: {n_scatter} -> {scatter_dir}")
    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time (s): {elapsed:.2f}")


if __name__ == "__main__":
    main()
