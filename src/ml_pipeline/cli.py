"""Windows-friendly commands for the complete ML workflow."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from ml_pipeline.data import load_csv_dataset, load_dataset
from ml_pipeline.evaluation import evaluate_model
from ml_pipeline.prediction import PredictionService, load_artifact
from ml_pipeline.reporting import render_existing_report
from ml_pipeline.training import TrainingConfig, train_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Tune four classifiers and save the CV winner")
    train.add_argument("--output-dir", type=Path, default=Path("."))
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--cv-folds", type=int, default=5)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--n-jobs", type=int, default=1)
    train.add_argument("--data", type=Path, help="Optional binary 0/1 training CSV")
    train.add_argument("--target", default="target")
    evaluate = commands.add_parser("evaluate", help="Re-evaluate the saved model on its holdout")
    evaluate.add_argument("--model", type=Path, default=Path("models/best_model.joblib"))
    evaluate.add_argument("--data", type=Path, default=Path("models/holdout.csv"))
    evaluate.add_argument("--output", type=Path, default=Path("reports/reevaluation.json"))
    report = commands.add_parser("report", help="Regenerate plots from saved evaluation JSON")
    report.add_argument("--reports-dir", type=Path, default=Path("reports"))
    predict = commands.add_parser("predict", help="Predict records from JSON or a feature CSV")
    predict.add_argument("--model", type=Path, default=Path("models/best_model.joblib"))
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path)
    sample = commands.add_parser("sample", help="Write a valid built-in dataset inference example")
    sample.add_argument("--output", type=Path, default=Path("examples/sample_input.json"))
    serve = commands.add_parser("serve", help="Start the inference API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model", type=Path, default=Path("models/best_model.joblib"))
    return parser


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            dataset = load_csv_dataset(args.data, args.target) if args.data else load_dataset()
            result = train_models(
                args.output_dir,
                TrainingConfig(args.test_size, args.cv_folds, args.seed, args.n_jobs),
                dataset,
            )
            print(
                json.dumps(
                    {
                        "best_model_name": result["best_model_name"],
                        "metrics": result["results"][result["best_model_name"]],
                    },
                    indent=2,
                )
            )
        elif args.command == "evaluate":
            artifact = load_artifact(args.model)
            if (
                hashlib.sha256(args.data.read_bytes()).hexdigest()
                != artifact["metadata"]["holdout_sha256"]
            ):
                raise ValueError("Holdout checksum mismatch: use the CSV from this training run")
            data = pd.read_csv(args.data, float_precision="round_trip")
            y = data.pop("__target__")
            metrics = evaluate_model(artifact["pipeline"], data, y)
            write_json(args.output, metrics)
            print(json.dumps(metrics, indent=2))
        elif args.command == "report":
            render_existing_report(args.reports_dir)
            print(f"Reports regenerated under {args.reports_dir}")
        elif args.command == "sample":
            write_json(
                args.output, {"instances": load_dataset().X.head(2).to_dict(orient="records")}
            )
            print(f"Sample written to {args.output}")
        elif args.command == "predict":
            if args.input.suffix.lower() == ".csv":
                frame = pd.read_csv(args.input)
                rows = frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
            else:
                rows = json.loads(args.input.read_text(encoding="utf-8"))["instances"]
            result = PredictionService(args.model).predict(rows)
            if args.output:
                write_json(args.output, result)
            print(json.dumps(result, indent=2))
        elif args.command == "serve":
            import uvicorn

            from ml_pipeline.api import create_app

            uvicorn.run(create_app(args.model), host=args.host, port=args.port)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
