import argparse
import json
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from dataset_deployment.scripts.pipeline import resolve_studies, run_pipeline_for_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full SurvPGC dataset generation pipeline.")
    parser.add_argument("--study", type=str, default=None)
    parser.add_argument("--all", action="store_true", dest="run_all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-gdc-api", action="store_true")
    parser.add_argument("--gene-embedding", default="scFoundation_embedding_gene_raw")
    parser.add_argument("--wsi-experiment", default="uni_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = {}
    for study in resolve_studies(args.study, args.run_all):
        summaries[study] = run_pipeline_for_study(
            study,
            dry_run=args.dry_run,
            validate_only=args.validate_only,
            allow_gdc_api=args.allow_gdc_api,
            gene_embedding=args.gene_embedding,
            wsi_experiment=args.wsi_experiment,
        )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
