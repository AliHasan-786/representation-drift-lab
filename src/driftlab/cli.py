from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import read_json, validate_web_artifact
from .benchmark import run_clip_benchmark_suite
from .config import load_config
from .clip_reproduction import regenerate_clip_metrics, run_clip_reproduction
from .experiment import run_smoke_experiment
from .early_warning import build_synthetic_methodology_artifact
from .legacy import verify_legacy_manifest
from .legacy_export import export_legacy_web_artifact
from .interpolation import run_interpolation_suite
from .method_benchmark import build_method_comparison
from .linear_probe import run_linear_probe_suite
from .wise_ft import run_wise_ft_suite
from .domain_benchmark import build_domain_comparison


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="driftlab", description="Representation Drift Lab experiment tools"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="run the deterministic CPU smoke experiment")
    smoke.add_argument("--config", default="configs/smoke.yaml")
    smoke.add_argument("--resume", action="store_true")
    smoke.add_argument("--web-output", default="public/data/smoke.json")
    validate = subparsers.add_parser(
        "validate-artifact", help="validate a public experiment artifact"
    )
    validate.add_argument("path")
    legacy = subparsers.add_parser(
        "verify-legacy", help="verify external legacy artifacts against checksums"
    )
    legacy.add_argument("--manifest", default="data/legacy/manifest.json")
    legacy.add_argument("--root")
    export_legacy = subparsers.add_parser(
        "export-legacy", help="export registered historical metrics for the web"
    )
    export_legacy.add_argument("--root", required=True)
    export_legacy.add_argument("--manifest", default="data/legacy/manifest.json")
    export_legacy.add_argument("--output", default="public/data/legacy-historical.json")
    reproduce = subparsers.add_parser(
        "reproduce-clip", help="run the CLIP Food-101 to CIFAR-10 reproduction"
    )
    reproduce.add_argument("--config", default="configs/reproduction-local.yaml")
    reproduce.add_argument("--resume", action="store_true")
    reproduce.add_argument("--web-output", default="public/data/reproduction-local.json")
    benchmark = subparsers.add_parser(
        "benchmark-clip", help="run and aggregate an independent-seed CLIP suite"
    )
    benchmark.add_argument(
        "--suite", default="configs/reproduction-local-multiseed.yaml"
    )
    benchmark.add_argument("--no-resume", action="store_true")
    benchmark.add_argument("--regenerate-metrics", action="store_true")
    regenerate = subparsers.add_parser(
        "regenerate-clip-metrics",
        help="recompute CLIP diagnostics from saved model outputs",
    )
    regenerate.add_argument("--config", default="configs/reproduction-local.yaml")
    regenerate.add_argument(
        "--web-output", default="public/data/reproduction-local.json"
    )
    early_warning = subparsers.add_parser(
        "early-warning-demo",
        help="build the explicitly synthetic held-out methodology artifact",
    )
    early_warning.add_argument(
        "--output", default="public/data/early-warning-methodology.json"
    )
    methods = subparsers.add_parser(
        "aggregate-methods", help="build a provenance-linked multi-method artifact"
    )
    methods.add_argument(
        "--config", default="configs/method-comparison-local.yaml"
    )
    interpolation = subparsers.add_parser(
        "interpolate-lora", help="evaluate a post-hoc LoRA output-scaling curve"
    )
    interpolation.add_argument("--config", default="configs/interpolation-local.yaml")
    linear_probe = subparsers.add_parser(
        "classification-suite", help="run a linear-probe, full-FT, or LP-FT suite"
    )
    linear_probe.add_argument(
        "--suite", default="configs/linear-probe-local-multiseed.yaml"
    )
    wise_ft = subparsers.add_parser(
        "wise-ft", help="run full-model weight-space ensemble evaluation"
    )
    wise_ft.add_argument("--config", default="configs/wise-ft-local.yaml")
    domains = subparsers.add_parser(
        "aggregate-domains", help="build the provenance-linked domain comparison"
    )
    domains.add_argument("--config", default="configs/domain-comparison-local.yaml")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "smoke":
        outputs = run_smoke_experiment(
            load_config(args.config), resume=args.resume, web_output=args.web_output
        )
        print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
        return
    if args.command == "validate-artifact":
        path = Path(args.path)
        validate_web_artifact(read_json(path))
        print(f"valid: {path}")
        return
    if args.command == "verify-legacy":
        result = verify_legacy_manifest(args.manifest, root=args.root)
        print(json.dumps(result, indent=2))
        if not result["all_valid"]:
            raise SystemExit(1)
        return
    if args.command == "export-legacy":
        output = export_legacy_web_artifact(
            root=args.root, output=args.output, manifest_path=args.manifest
        )
        validate_web_artifact(read_json(output))
        print(f"exported and valid: {output}")
        return
    if args.command == "reproduce-clip":
        output = run_clip_reproduction(
            load_config(args.config),
            resume=args.resume,
            web_output=args.web_output,
        )
        print(json.dumps({key: str(value) for key, value in output.items()}, indent=2))
        return
    if args.command == "benchmark-clip":
        output = run_clip_benchmark_suite(
            args.suite,
            resume=not args.no_resume,
            regenerate_metrics=args.regenerate_metrics,
        )
        print(json.dumps({key: str(value) for key, value in output.items()}, indent=2))
        return
    if args.command == "regenerate-clip-metrics":
        output = regenerate_clip_metrics(
            load_config(args.config), web_output=args.web_output
        )
        print(json.dumps({key: str(value) for key, value in output.items()}, indent=2))
        return
    if args.command == "early-warning-demo":
        output = build_synthetic_methodology_artifact(args.output)
        print(f"generated: {output}")
        return
    if args.command == "aggregate-methods":
        output = build_method_comparison(args.config)
        print(f"generated: {output}")
        return
    if args.command == "interpolate-lora":
        output = run_interpolation_suite(args.config)
        print(f"generated: {output}")
        return
    if args.command == "classification-suite":
        output = run_linear_probe_suite(args.suite)
        print(f"generated: {output}")
        return
    if args.command == "wise-ft":
        output = run_wise_ft_suite(args.config)
        print(f"generated: {output}")
        return
    if args.command == "aggregate-domains":
        output = build_domain_comparison(args.config)
        print(f"generated: {output}")
        return
    raise AssertionError(f"unhandled command: {args.command}")
