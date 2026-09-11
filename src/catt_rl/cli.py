"""Command-line interface for the complete CATT-RL workflow."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from catt_rl.config import apply_mapping_overrides, load_config
from catt_rl.data.download import download_yahoo
from catt_rl.data.features import prepare_panel_file
from catt_rl.data.panel import MarketPanel
from catt_rl.data.qlib_alpha158 import export_alpha158
from catt_rl.data.splits import load_protocol
from catt_rl.data.synthetic import make_synthetic_panel
from catt_rl.evaluation import evaluate_policy, load_checkpoint_model
from catt_rl.experiment import build_model, train_experiment
from catt_rl.utils import resolve_device
from catt_rl.visualization import plot_attention, plot_training

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def _config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Experiment YAML")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable dotted configuration override",
    )


def command_synthetic(args: argparse.Namespace) -> None:
    path = make_synthetic_panel(args.output, args.assets, args.days, args.seed, args.start)
    print(
        f"Wrote synthetic panel: {args.output} ({path.num_dates} dates, {path.num_assets} assets)"
    )


def command_download(args: argparse.Namespace) -> None:
    output = download_yahoo(args.tickers, args.start, args.end, args.output, not args.no_threads)
    print(f"Wrote raw OHLCV: {output}")


def command_prepare(args: argparse.Namespace) -> None:
    selected = None
    if args.alpha158_columns:
        selected = [
            line.strip()
            for line in Path(args.alpha158_columns).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    output = prepare_panel_file(
        args.raw,
        args.output,
        availability=args.availability,
        feature_set=args.feature_set,
        warmup=args.warmup,
        availability_start=args.availability_start,
        availability_end=args.availability_end,
        external_factors=args.alpha158,
        selected_external_columns=selected,
    )
    print(f"Wrote prepared panel: {output}")


def command_train(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.overrides)
    run_dir = train_experiment(config)
    print(f"Completed run: {run_dir}")


def command_evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.overrides)
    panel = MarketPanel.load(config["data"]["panel_path"]).select_features(
        str(config["data"]["feature_set"])
    )
    device = resolve_device(str(config["experiment"]["device"]))
    model = load_checkpoint_model(args.checkpoint, panel, config, device)
    if args.split == "validation":
        start, end = config["data"]["validation_start"], config["data"]["validation_end"]
    else:
        start, end = config["data"]["test_start"], config["data"]["test_end"]
    output = (
        Path(args.output)
        if args.output
        else Path(args.checkpoint).parent / f"{args.split}_evaluation"
    )
    result = evaluate_policy(
        model,
        panel,
        config,
        str(start),
        str(end),
        device,
        output_dir=output,
        bootstrap=not args.no_bootstrap,
        save_attention=args.attention,
    )
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


def command_inspect(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.overrides)
    panel = MarketPanel.load(config["data"]["panel_path"]).select_features(
        str(config["data"]["feature_set"])
    )
    model = build_model(panel, config, torch.device("cpu"))
    summary = {
        "dates": [str(panel.dates[0]), str(panel.dates[-1])],
        "assets": panel.num_assets,
        "features": panel.num_features,
        "feature_names": list(panel.feature_names),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "memory_shape_per_environment": [model.memory_depth, model.dimension],
    }
    print(json.dumps(summary, indent=2))


def command_multiseed(args: argparse.Namespace) -> None:
    base = load_config(args.config, args.overrides)
    completed: list[str] = []
    for seed in args.seeds:
        config = copy.deepcopy(base)
        config["experiment"]["seed"] = seed
        if args.dry_run:
            completed.append(f"seed={seed}")
        else:
            completed.append(str(train_experiment(config)))
    print("\n".join(completed))


def _ablation_files(group: str) -> list[Path]:
    directory = REPOSITORY_ROOT / "configs" / "ablations"
    if group == "all":
        return sorted(directory.glob("*.yaml"))
    aliases = {"norm": "normalization_gating"}
    name = aliases.get(group, group)
    path = directory / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    return [path]


def command_ablate(args: argparse.Namespace) -> None:
    base = load_config(args.config, args.overrides)
    planned: list[dict[str, Any]] = []
    for path in _ablation_files(args.group):
        specification = yaml.safe_load(path.read_text(encoding="utf-8"))
        group = specification["group"]
        for variant in specification["variants"]:
            for seed in args.seeds:
                config = apply_mapping_overrides(base, variant["overrides"])
                config["experiment"]["seed"] = seed
                config["experiment"]["name"] = f"{base['experiment']['name']}_{variant['name']}"
                config["experiment"]["output_dir"] = str(
                    Path(base["experiment"]["output_dir"]) / "ablations" / group / variant["name"]
                )
                item = {"group": group, "variant": variant["name"], "seed": seed}
                if not args.dry_run:
                    item["run_dir"] = str(train_experiment(config))
                planned.append(item)
    print(json.dumps(planned, indent=2))


def command_protocol(args: argparse.Namespace) -> None:
    base = load_config(args.config, args.overrides)
    protocol = load_protocol(args.protocol)
    completed: list[dict[str, Any]] = []
    for window in protocol["windows"]:
        for seed in args.seeds:
            config = copy.deepcopy(base)
            config["experiment"]["seed"] = seed
            config["experiment"]["name"] = f"{base['experiment']['name']}_{window['name']}"
            config["experiment"]["output_dir"] = str(
                Path(base["experiment"]["output_dir"])
                / "protocols"
                / protocol["name"]
                / window["name"]
            )
            for key in (
                "panel_path",
                "train_start",
                "train_end",
                "validation_start",
                "validation_end",
                "test_start",
                "test_end",
            ):
                if key in window:
                    config["data"][key] = window[key]
            item = {"window": window["name"], "seed": seed}
            if not args.dry_run:
                item["run_dir"] = str(train_experiment(config))
            completed.append(item)
    print(json.dumps(completed, indent=2))


def command_aggregate(args: argparse.Namespace) -> None:
    root = Path(args.input)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("test_metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        row = {"run": str(path.parent.relative_to(root))}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) or value is None:
                row[key] = value
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No test_metrics.json files under {root}")
    columns = sorted({key for row in rows for key in row if key != "run"})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", *columns])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows: {output}")


def command_qlib(args: argparse.Namespace) -> None:
    output = export_alpha158(
        args.provider_uri,
        args.instruments,
        args.start,
        args.end,
        args.output,
        args.columns,
    )
    print(f"Wrote Alpha158 factors: {output}")


def command_plot(args: argparse.Namespace) -> None:
    if args.kind == "attention":
        output = plot_attention(args.input, args.output, args.key)
    else:
        output = plot_training(args.input, args.output)
    print(f"Wrote figure: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="catt-rl", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic", help="Create deterministic smoke data")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--assets", type=int, default=4)
    synthetic.add_argument("--days", type=int, default=220)
    synthetic.add_argument("--seed", type=int, default=7)
    synthetic.add_argument("--start", default="2018-01-01")
    synthetic.set_defaults(function=command_synthetic)

    download = subparsers.add_parser("download", help="Download permitted Yahoo OHLCV")
    download.add_argument("--tickers", required=True, help="CSV with ticker column")
    download.add_argument("--start", required=True)
    download.add_argument("--end", required=True, help="Exclusive end date")
    download.add_argument("--output", required=True)
    download.add_argument("--no-threads", action="store_true")
    download.set_defaults(function=command_download)

    prepare = subparsers.add_parser("prepare", help="Create a leakage-safe panel")
    prepare.add_argument("--raw", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--availability", type=float, default=0.80)
    prepare.add_argument("--availability-start")
    prepare.add_argument("--availability-end")
    prepare.add_argument(
        "--feature-set", choices=["prices", "momentum", "volatility_range", "full"], default="full"
    )
    prepare.add_argument("--warmup", type=int, default=126)
    prepare.add_argument("--alpha158", help="Optional exported Alpha158 long CSV")
    prepare.add_argument("--alpha158-columns", help="One exact factor column per line")
    prepare.set_defaults(function=command_prepare)

    train = subparsers.add_parser("train", help="Train one seed/split")
    _config_arguments(train)
    train.set_defaults(function=command_train)

    evaluate = subparsers.add_parser("evaluate", help="Backtest a checkpoint")
    _config_arguments(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    evaluate.add_argument("--output")
    evaluate.add_argument("--no-bootstrap", action="store_true")
    evaluate.add_argument("--attention", action="store_true")
    evaluate.set_defaults(function=command_evaluate)

    inspect = subparsers.add_parser("inspect", help="Validate panel/config and show shapes")
    _config_arguments(inspect)
    inspect.set_defaults(function=command_inspect)

    multiseed = subparsers.add_parser("multiseed", help="Run independent seeds")
    _config_arguments(multiseed)
    multiseed.add_argument("--seeds", type=_seeds, required=True)
    multiseed.add_argument("--dry-run", action="store_true")
    multiseed.set_defaults(function=command_multiseed)

    ablate = subparsers.add_parser("ablate", help="Run manuscript ablations")
    _config_arguments(ablate)
    ablate.add_argument(
        "--group",
        default="all",
        choices=["all", "backbone", "memory", "normalization_gating", "norm", "reward", "features"],
    )
    ablate.add_argument("--seeds", type=_seeds, default=[7])
    ablate.add_argument("--dry-run", action="store_true")
    ablate.set_defaults(function=command_ablate)

    protocol = subparsers.add_parser("protocol", help="Run explicit rolling windows")
    _config_arguments(protocol)
    protocol.add_argument("--protocol", required=True)
    protocol.add_argument("--seeds", type=_seeds, default=[7])
    protocol.add_argument("--dry-run", action="store_true")
    protocol.set_defaults(function=command_protocol)

    aggregate = subparsers.add_parser("aggregate", help="Collect generated test metrics")
    aggregate.add_argument("--input", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.set_defaults(function=command_aggregate)

    qlib = subparsers.add_parser("qlib-alpha158", help="Export official Qlib Alpha158 factors")
    qlib.add_argument("--provider-uri", required=True)
    qlib.add_argument("--instruments", required=True)
    qlib.add_argument("--start", required=True)
    qlib.add_argument("--end", required=True)
    qlib.add_argument("--output", required=True)
    qlib.add_argument("--columns", help="Optional exact factor subset file")
    qlib.set_defaults(function=command_qlib)

    plot = subparsers.add_parser("plot", help="Plot attention diagnostics or training")
    plot.add_argument("--kind", choices=["attention", "training"], required=True)
    plot.add_argument("--input", required=True)
    plot.add_argument("--output", required=True)
    plot.add_argument("--key", default="head_feature_projection")
    plot.set_defaults(function=command_plot)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
