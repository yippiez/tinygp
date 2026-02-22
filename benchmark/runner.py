import argparse

from benchmark.common import benchmark_gplearn
from benchmark.common import benchmark_strategy
from benchmark.targets import TARGET_REGISTRY

from tinygp.strategies import STRATEGY_REGISTRY


def ordered_strategy_names() -> list[str]:
    """Return strategy names with BasicStrategy first."""
    if "BasicStrategy" not in STRATEGY_REGISTRY:
        return sorted(STRATEGY_REGISTRY)
    return ["BasicStrategy"] + sorted(name for name in STRATEGY_REGISTRY if name != "BasicStrategy")


def parse_args(default_strategy: str | None = None) -> argparse.Namespace:
    """Parse command-line arguments for benchmark runs."""
    if default_strategy is not None:
        assert default_strategy in STRATEGY_REGISTRY, "default strategy must be a registered strategy"

    strategy_choices = ["all"] + ordered_strategy_names()
    target_choices = ["all"] + sorted(TARGET_REGISTRY)

    parser = argparse.ArgumentParser(description="Run tinygp strategy benchmarks")
    parser.add_argument("--strategy", choices=strategy_choices, default=default_strategy or "all")
    parser.add_argument("--target", choices=target_choices, default="all")
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--simplify-every-n", type=int, default=0)
    parser.add_argument("--skip-gplearn", action="store_true")
    parser.add_argument("--show-expr", action="store_true")
    parser.add_argument("--list-strategies", action="store_true")
    parser.add_argument("--list-targets", action="store_true")
    return parser.parse_args()


def selected_strategies(strategy_arg: str) -> list[str]:
    """Resolve the strategy selection argument into names."""
    if strategy_arg == "all":
        return ordered_strategy_names()
    return [strategy_arg]


def selected_targets(target_arg: str) -> list[str]:
    """Resolve the target selection argument into names."""
    if target_arg == "all":
        return sorted(TARGET_REGISTRY)
    return [target_arg]


def run_cli(default_strategy: str | None = None) -> None:
    """Run the benchmark CLI with optional default strategy."""
    args = parse_args(default_strategy=default_strategy)

    if args.list_strategies:
        for name in ordered_strategy_names():
            print(name)
        return

    if args.list_targets:
        for name in sorted(TARGET_REGISTRY):
            print(name)
        return

    assert args.generations > 0, "generations must be positive"
    assert args.simplify_every_n >= 0, "simplify_every_n must be non-negative"

    strategy_names = selected_strategies(args.strategy)
    target_names = selected_targets(args.target)

    for target_name in target_names:
        print(f"benchmark: {target_name}")
        print("method                 train_mse    test_mse     test_mae     iter_sec")

        rows: list[dict] = []
        for strategy_name in strategy_names:
            row = benchmark_strategy(
                strategy_name,
                target_name,
                args.generations,
                simplify_every_n=args.simplify_every_n,
            )
            rows.append(row)
            print(
                f"{row['method']:20s} {row['train_mse']:10.6f}  {row['test_mse']:10.6f}  "
                f"{row['test_mae']:10.6f}  {row['iter_sec']:9.5f}"
            )

        gplearn_row: dict | None = None
        if not args.skip_gplearn:
            gplearn_row = benchmark_gplearn(target_name, args.generations)
            print(
                f"{'gplearn':20s} {gplearn_row['train_mse']:10.6f}  {gplearn_row['test_mse']:10.6f}  "
                f"{gplearn_row['test_mae']:10.6f}  {gplearn_row['iter_sec']:9.5f}"
            )

        if args.show_expr:
            for row in rows:
                print(f"{row['method']} expr: {row['expr']}")
            if gplearn_row is not None:
                print(f"gplearn expr: {gplearn_row['expr']}")

        print()


def main() -> None:
    """Run the benchmark runner entrypoint."""
    run_cli()


if __name__ == "__main__":
    main()
