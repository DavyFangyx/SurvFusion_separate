"""
Generate rna_clean.csv for KICH / KIRC / KIRP from TCGA Bulk_RNA TSV files.

For each cancer subtype:
    1. Read A_both_{SUBTYPE}.csv to get patient_id -> rna_file_name mapping.
    2. Locate the .rna_seq.augmented_star_gene_counts.tsv inside Bulk_RNA/{rna_file_name}/.
    3. Extract tpm_unstranded for protein_coding genes.
    4. Build a raw (patients x genes) matrix.
    5. Reindex genes to the combine_signatures gene order, filling missing genes with 0.
    6. Write a fixed-width rna_clean.csv with 4999 genes.

Usage:
python generate_rna_clean_csv.py
        python generate_rna_clean_csv.py --subtype kich        # generate only kich
        python generate_rna_clean_csv.py --bulk-rna-dir /path  # override bulk_rna directory
"""

import argparse
import os
from glob import glob

import pandas as pd

DEFAULT_BULK_RNA_DIR = "/data/lizhe/Medteam_projects/kindey_cancer_TCGA/Bulk_RNA"
DEFAULT_INDEX_DIR = "/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/patients_index"
DEFAULT_OUTPUT_BASE = "/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/datasets_csv/raw_rna_data/combine"
DEFAULT_SIGNATURE_CSV = "/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/datasets_csv/metadata/combine_signatures.csv"

SUBTYPE_CONFIG = {
    "kich": {"index_csv": "A_both_KICH.csv"},
    "kirc": {"index_csv": "A_both_KIRC.csv"},
    "kirp": {"index_csv": "A_both_KIRP.csv"},
}

# Rows in the TSV that are summary statistics, not genes
STAT_ROW_PREFIXES = ("N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pathway-level gene expression matrix (rna_clean.csv) for kidney cancer subtypes."
    )
    parser.add_argument("--bulk-rna-dir", default=DEFAULT_BULK_RNA_DIR,
                        help="Root directory containing per-sample RNA-seq subdirectories.")
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR,
                        help="Directory containing A_both_*.csv index files.")
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE,
                        help="Base output directory (will write {subtype}/rna_clean.csv).")
    parser.add_argument("--signature-csv", default=DEFAULT_SIGNATURE_CSV,
                        help="combine_signatures.csv used to define output gene order.")
    parser.add_argument("--subtype", choices=sorted(SUBTYPE_CONFIG.keys()), default=None,
                        help="Generate only this subtype; omit to generate all.")
    return parser.parse_args()


def load_target_genes(signature_csv: str) -> list[str]:
    """
    Read combine_signatures.csv and return the deduplicated target gene list.

    Genes are traversed column-by-column in the original CSV order so the generated
    rna_clean.csv has a stable and reproducible 4999-gene layout.
    """
    signatures_df = pd.read_csv(signature_csv)
    ordered_genes = []
    seen_genes = set()

    for column in signatures_df.columns:
        for value in signatures_df[column].dropna():
            gene = str(value).strip()
            if not gene or gene in seen_genes:
                continue
            seen_genes.add(gene)
            ordered_genes.append(gene)

    return ordered_genes


def load_patient_index(index_csv: str) -> pd.DataFrame:
    """Read patient index CSV (patient_id, rna_file_name, ...)."""
    df = pd.read_csv(index_csv)
    # Normalise header names
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
    return df


def find_tsv(bulk_rna_dir: str, rna_file_name: str) -> str | None:
    """Find the single .tsv file inside bulk_rna_dir/rna_file_name/."""
    sample_dir = os.path.join(bulk_rna_dir, rna_file_name)
    if not os.path.isdir(sample_dir):
        return None
    tsv_files = glob(os.path.join(sample_dir, "*.tsv"))
    if len(tsv_files) == 1:
        return tsv_files[0]
    # fallback: search for the gene_counts tsv specifically
    candidates = [f for f in tsv_files if "gene_counts" in os.path.basename(f)]
    return candidates[0] if candidates else None


def read_tpm_from_tsv(tsv_path: str) -> pd.Series:
    """
    Read a single TCGA RNA-seq TSV and return a Series:
        index = gene_name, value = tpm_unstranded
    Only protein_coding genes are kept; duplicates are summed.
    """
    df = pd.read_csv(tsv_path, sep="\t", comment="#")
    # drop summary / stat rows
    df = df[~df["gene_id"].isin(STAT_ROW_PREFIXES)]
    # keep protein_coding only
    df = df[df["gene_type"] == "protein_coding"]
    # aggregate duplicate gene_names (rare, but possible with different ENSG ids)
    tpm = df.groupby("gene_name")["tpm_unstranded"].sum()
    return tpm


def build_expression_matrix(
    patient_df: pd.DataFrame,
    bulk_rna_dir: str,
    target_genes: list[str],
) -> pd.DataFrame:
    """
    For each patient, read TPM values and build a raw expression matrix.
    Returns a DataFrame with case_id as index and genes as columns.
    The final columns are forced to match the target gene order from combine_signatures,
    with any missing genes filled by 0.
    """
    records = {}
    missing = []

    for _, row in patient_df.iterrows():
        patient_id = row["patient_id"]
        rna_file_name = row["rna_file_name"]
        tsv_path = find_tsv(bulk_rna_dir, str(rna_file_name))
        if tsv_path is None:
            missing.append(patient_id)
            print(f"  WARNING: TSV not found for {patient_id} (dir={rna_file_name})")
            continue

        tpm = read_tpm_from_tsv(tsv_path)
        records[patient_id] = tpm

    if missing:
        print(f"  {len(missing)} patients skipped (no TSV found).")

    # build DataFrame
    expr_df = pd.DataFrame.from_dict(records, orient="index")
    expr_df = expr_df.reindex(columns=target_genes, fill_value=0.0)
    expr_df = expr_df.fillna(0.0)
    expr_df.index.name = "case_id"
    return expr_df


def generate_for_subtype(
    subtype: str,
    bulk_rna_dir: str,
    index_dir: str,
    output_base: str,
    target_genes: list[str],
):
    cfg = SUBTYPE_CONFIG[subtype]
    index_csv = os.path.join(index_dir, cfg["index_csv"])
    print(f"[{subtype.upper()}] Loading patient index from {index_csv}")
    patient_df = load_patient_index(index_csv)
    print(f"  {len(patient_df)} patients found")

    print(f"  Building expression matrix ...")
    expr_df = build_expression_matrix(patient_df, bulk_rna_dir, target_genes)
    print(f"  Matrix shape: {expr_df.shape}")

    output_dir = os.path.join(output_base, subtype)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rna_clean.csv")
    expr_df.to_csv(output_path)
    print(f"  Saved to {output_path}")


def main():
    args = parse_args()
    target_genes = load_target_genes(args.signature_csv)
    print(f"Loaded {len(target_genes)} target genes from {args.signature_csv}")

    subtypes = [args.subtype] if args.subtype else sorted(SUBTYPE_CONFIG.keys())
    for subtype in subtypes:
        generate_for_subtype(
            subtype=subtype,
            bulk_rna_dir=args.bulk_rna_dir,
            index_dir=args.index_dir,
            output_base=args.output_base,
            target_genes=target_genes,
        )
    print("Done.")


if __name__ == "__main__":
    main()
