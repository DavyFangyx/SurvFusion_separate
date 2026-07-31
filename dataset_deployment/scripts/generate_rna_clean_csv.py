import argparse
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from dataset_deployment.scripts.pipeline import generate_rna_for_study, resolve_studies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rna_clean.csv for one or more SurvPGC datasets.")
    parser.add_argument("--study", type=str, default=None)
    parser.add_argument("--all", action="store_true", dest="run_all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-gdc-api", action="store_true")
    parser.add_argument("--signature-csv", default="datasets_csv/metadata/combine_signatures.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    for study in resolve_studies(args.study, args.run_all):
        print(
            generate_rna_for_study(
                study,
                dry_run=args.dry_run,
                validate_only=args.validate_only,
                allow_gdc_api=args.allow_gdc_api,
                signature_csv=args.signature_csv,
            )
        )


if __name__ == "__main__":
    main()
