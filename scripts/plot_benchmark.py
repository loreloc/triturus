import argparse
import json
import os
import subprocess
import textwrap
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from triton.testing import Benchmark, Mark

from benchmarks.add import benchmark_vadd
from benchmarks.lm2exp import benchmark_lm2exp
from benchmarks.max import benchmark_matmax, benchmark_vmax
from benchmarks.mm import benchmark_mm

BENCHMARKS: Mapping[str, Mark] = {
    "vadd": benchmark_vadd,
    "vmax": benchmark_vmax,
    "matmax": benchmark_matmax,
    "mm": benchmark_mm,
    "lm2exp": benchmark_lm2exp,
}


parser = argparse.ArgumentParser(description="Triturus -- benchmark results plotter")
parser.add_argument(
    "--names",
    nargs="+",
    type=str,
    required=True,
    help="The name(s) of the benchmarks to plot. Use '*' to plot all of them.",
)
parser.add_argument(
    "--print-results",
    action="store_true",
    default=False,
    help="Whether to also print to stdout the benchmark results.",
)
parser.add_argument(
    "--allow-tf32",
    action="store_true",
    default=False,
    help="Whether to only plot results obtained with tf32, "
    "if it is something specified for a benchmark",
)


def git_hash(short: bool = True) -> str:
    cmd = ["git", "rev-parse"]
    if short:
        cmd.append("--short")
    cmd.append("HEAD")
    return subprocess.check_output(cmd).decode("ascii").strip()


@dataclass
class MarkSchema:
    providers: list[str]
    x_names: list[str]
    fixed_args: dict[str, Any]
    metric: str


class MarkRunner:
    def __init__(self, mark: Mark) -> None:
        self._mark = mark

    def _run(self, bench: Benchmark, **kwargs: Any) -> tuple[pd.DataFrame, MarkSchema]:
        rows = defaultdict(list)
        x_names = list(bench.x_names)
        for x in bench.x_vals:
            if not isinstance(x, (list, tuple)):
                x = [x for _ in x_names]
            if len(x) != len(x_names):
                raise ValueError(f"Expected {len(x_names)} values, got {x}")
            for x_name, x_val in zip(x_names, x):
                rows[x_name].append(x_val)
            x_args = dict(zip(x_names, x))
            for y in bench.line_vals:
                results = self._mark.fn(**x_args, **{bench.line_arg: y}, **bench.args, **kwargs)
                y_mean, y_min, y_max = results
                rows[y].append(y_mean)
                rows[f"{y}-min"].append(y_min)
                rows[f"{y}-max"].append(y_max)
        df = pd.DataFrame(rows)
        schema = MarkSchema(
            providers=bench.line_vals,
            x_names=x_names,
            fixed_args=bench.args,
            metric=bench.ylabel,
        )
        return df, schema

    def run(self, **kwargs: Any) -> list[tuple[pd.DataFrame, MarkSchema]]:
        benchmarks: list[Benchmark]
        if isinstance(self._mark.benchmarks, Benchmark):
            benchmarks = [self._mark.benchmarks]
        else:
            benchmarks = self._mark.benchmarks
        dfs_schemas: list[tuple[pd.DataFrame, MarkSchema]] = []
        for bench in benchmarks:
            dfs_schemas.append(self._run(bench, **kwargs))
        return dfs_schemas


def plot_benchmark_result(
    df: pd.DataFrame,
    args: Mapping[str, Any],
    providers: Sequence[str],
    x_names: Sequence[str],
    y_label: str,
    *,
    title: str,
    show_y_label: bool = True,
    show_legend: bool = True,
    ax: plt.Axes,
) -> None:
    ax.grid(linestyle="--", which="major", alpha=0.35, linewidth=0.5)
    ax.grid(linestyle="--", which="minor", alpha=0.35, linewidth=0.3)
    x_col = x_names[0]
    for provider in providers:
        y_min, y_max = df[provider + "-min"], df[provider + "-max"]
        ax.fill_between(df[x_col], y_min, y_max, alpha=0.2)
        ax.plot(df[x_col], df[provider], label=provider, alpha=0.8)
    if show_legend:
        ax.legend()
    formatted_args = dict((k if len(k) > 1 else f"${k}$", v) for k, v in args.items())
    fixed_args_repr = ", ".join(f"{k}={v}" for k, v in formatted_args.items())
    if fixed_args_repr:
        title = f"{title} ({fixed_args_repr})"
    title_wrapped = "\n".join(textwrap.wrap(title, width=36))
    ax.set_title(title_wrapped)
    ax.set_xscale("log")
    x_labels = [x if len(x) > 1 else f"${x}$" for x in x_names]
    ax.set_xlabel(", ".join(x_labels))
    if show_y_label:
        ax.set_ylabel(y_label)


if __name__ == "__main__":
    args = parser.parse_args()
    names: list[str] = args.names
    if len(names) == 1 and names[0] == "*":
        names = list(BENCHMARKS.keys())
    allow_tf32: bool = args.allow_tf32
    print_results: bool = args.print_results
    results_path: str = f"results-{git_hash()}"
    for name in names:
        if name not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark name '{name}'")
    os.makedirs(results_path, exist_ok=True)

    # Run benchmarks, retrieve the results and serialize them
    for name in names:
        filepath = os.path.join(results_path, f"{name}.json")
        if os.path.isfile(filepath):
            print(f"Using cached '{name}' benchmark results at {filepath} ...")
            continue
        mark = BENCHMARKS[name]
        benchmarks = mark.benchmarks
        print(f"Running benchmark '{name}' ...")
        dfs_schemas = MarkRunner(mark).run()
        data_json: list[dict[str, Any]] = []
        for df, schema in dfs_schemas:
            data_json.append({"schema": asdict(schema), "results": df.to_dict()})
        with open(filepath, "w") as fp:
            json.dump(data_json, fp)

    # Read the results and plot them
    for name in names:
        filepath = os.path.join(results_path, f"{name}.json")
        print(f"Loading benchmark '{name}' results from {filepath} ...")
        with open(filepath, "r") as fp:
            data_json = json.load(fp)
        dfs_schemas: list[tuple[pd.DataFrame, MarkSchema]] = []
        for i, row_json in enumerate(data_json):
            schema = MarkSchema(**row_json["schema"])
            df = pd.DataFrame(row_json["results"])
            if "allow_tf32" in schema.fixed_args:
                if schema.fixed_args["allow_tf32"] != allow_tf32:
                    continue
                del schema.fixed_args["allow_tf32"]
            if print_results:
                print(f"Benchmark #{i}:")
                if schema.fixed_args:
                    print(
                        "Fixed args:",
                        ", ".join(f"{k}={v}" for k, v in schema.fixed_args.items()),
                    )
                print(f"Metric: {schema.metric}")
                print("Results:")
                print(df)
            dfs_schemas.append((df, schema))
        filepath = os.path.join(results_path, f"{name}.pdf")
        print(f"Plotting benchmark '{name}' results to {filepath} ...")
        num_plots = len(dfs_schemas)
        figsize = (3.5, 3) if num_plots == 1 else (3 * num_plots, 3)
        fig, axs = plt.subplots(
            1,
            num_plots,
            figsize=figsize,
            layout="constrained",
            sharey=True,
            squeeze=False,
        )
        for i, (df, schema) in enumerate(dfs_schemas):
            plot_benchmark_result(
                df,
                schema.fixed_args,
                schema.providers,
                schema.x_names,
                schema.metric,
                title=name,
                show_y_label=i == 0,
                show_legend=i == 0,
                ax=axs[0, i],
            )
        fig.savefig(filepath, bbox_inches="tight")
        plt.clf()
