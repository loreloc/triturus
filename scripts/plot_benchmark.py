import argparse
import os
import subprocess
from collections.abc import Mapping, Sequence
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


class MarkWrapper:
    def __init__(self, mark: Mark):
        self._mark = mark

    def _run(
        self,
        bench: Benchmark,
        save_path: str,
        show_plots: bool,
        print_data: bool,
        diff_col=False,
        save_precision: int = 6,
        **kwargs: Any,
    ):
        y_mean = bench.line_names
        y_min = [f"{x}-min" for x in bench.line_names]
        y_max = [f"{x}-max" for x in bench.line_names]
        x_names = list(bench.x_names)
        df = pd.DataFrame(columns=x_names + y_mean + y_min + y_max)
        for x in bench.x_vals:
            if not isinstance(x, (list, tuple)):
                x = [x for _ in x_names]
            if len(x) != len(x_names):
                raise ValueError(f"Expected {len(x_names)} values, got {x}")
            x_args = dict(zip(x_names, x))
            row_mean, row_min, row_max = [], [], []
            for y in bench.line_vals:
                ret = self._mark.fn(
                    **x_args, **{bench.line_arg: y}, **bench.args, **kwargs
                )
                try:
                    y_mean, y_min, y_max = ret
                except TypeError:
                    y_mean, y_min, y_max = ret, None, None
                row_mean += [y_mean]
                row_min += [y_min]
                row_max += [y_max]
            df.loc[len(df)] = list(x) + row_mean + row_min + row_max
        if bench.plot_name:
            plt.figure()
            ax = plt.subplot()
            first_x = x_names[0]
            for i, y in enumerate(bench.line_names):
                y_min, y_max = df[y + "-min"], df[y + "-max"]
                col = bench.styles[i][0] if bench.styles else None
                sty = bench.styles[i][1] if bench.styles else None
                ax.plot(df[first_x], df[y], label=y, color=col, ls=sty)
                if not y_min.isnull().all() and not y_max.isnull().all():
                    y_min = y_min.astype(float)
                    y_max = y_max.astype(float)
                    ax.fill_between(df[first_x], y_min, y_max, alpha=0.15, color=col)
            ax.legend()
            ax.set_xlabel(bench.xlabel or first_x)
            ax.set_ylabel(bench.ylabel)
            ax.set_xscale("log" if bench.x_log else "linear")
            ax.set_yscale("log" if bench.y_log else "linear")
            if show_plots:
                plt.show()
            if save_path:
                plt.savefig(os.path.join(save_path, f"{bench.plot_name}.png"))
        # df = df[x_names + bench.line_names]
        if diff_col and df.shape[1] == 2:
            col0, col1 = df.columns.tolist()
            df["Diff"] = df[col1] - df[col0]
        if print_data:
            print(bench.plot_name + ":")
            print(df.to_string())
        if save_path:
            df.to_csv(
                os.path.join(save_path, f"{bench.plot_name}.csv"),
                float_format=f"%.{save_precision}f",
                index=False,
            )
        return df

    def run(
        self,
        show_plots: bool = False,
        print_data: bool = False,
        save_path: str = "",
        return_df: bool = False,
        **kwargs: Any,
    ):
        has_single_bench = isinstance(self._mark.benchmarks, Benchmark)
        benchmarks = (
            [self._mark.benchmarks] if has_single_bench else self._mark.benchmarks
        )
        result_dfs = []
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            html = open(os.path.join(save_path, "results.html"), "w")
            html.write("<html><body>\n")
        for bench in benchmarks:
            result_dfs.append(
                self._run(bench, save_path, show_plots, print_data, **kwargs)
            )
            if save_path:
                html.write(f'<image src="{bench.plot_name}.png"/>\n')
        if save_path:
            html.write("</body></html>\n")
            html.close()
        if return_df:
            if has_single_bench:
                return result_dfs[0]
            else:
                return result_dfs
        return None


def git_hash(short: bool = True) -> str:
    cmd = ["git", "rev-parse"]
    if short:
        cmd.append("--short")
    cmd.append("HEAD")
    return subprocess.check_output(cmd).decode("ascii").strip()


def plot_benchmark_result(
    df: pd.DataFrame,
    args: Mapping[str, Any],
    providers: Sequence[str],
    x_names: Sequence[str],
    y_label: str,
    title: str,
    *,
    show_y_label: bool = True,
    ax: plt.Axes,
) -> None:
    df = df.copy()
    for args_name, args_val in args.items():
        df[args_name] = args_val
    ax.grid(linestyle="--", which="major", alpha=0.3, linewidth=0.5)
    ax.grid(linestyle="--", which="minor", alpha=0.3, linewidth=0.3)
    x_col = x_names[0]
    for provider in providers:
        y_min, y_max = df[provider + "-min"], df[provider + "-max"]
        ax.fill_between(df[x_col], y_min, y_max, alpha=0.2)
        ax.plot(df[x_col], df[provider], label=provider)
    ax.legend()
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_xlabel(", ".join(x_names))
    if show_y_label:
        ax.set_ylabel(y_label)


if __name__ == "__main__":
    args = parser.parse_args()
    names: list[str] = args.names
    if len(names) == 1 and names[0] == "*":
        names = list(BENCHMARKS.keys())
    print_results: bool = args.print_results
    results_path: str = f"results-{git_hash()}"
    os.makedirs(results_path, exist_ok=True)
    for name in names:
        if name not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark name '{name}'")
        mark = BENCHMARKS[name]
        benchmarks = mark.benchmarks
        print(f"Running benchmark '{name}' ...")
        results: pd.DataFrame | Sequence[pd.DataFrame] = MarkWrapper(mark).run(
            print_data=print_results, return_df=True
        )
        dfs: list[pd.DataFrame] = (
            [results] if isinstance(results, pd.DataFrame) else list(results)
        )
        filepath = os.path.join(results_path, f"{name}.pdf")
        print(f"Plotting benchmark '{name}' results to {filepath} ...")
        figsize = (5, 4) if len(dfs) == 1 else (4 * len(dfs), 4)
        fig, axs = plt.subplots(
            1,
            len(dfs),
            figsize=figsize,
            layout="constrained",
            sharey=True,
            squeeze=False,
        )
        y_labels = set(b.ylabel for b in benchmarks)
        if len(y_labels) != 1:
            raise ValueError(
                f"Benchmark '{name}' runs should have the same metric"
                f" (i.e., the ylabel), but found {y_labels}"
            )
        (y_label,) = tuple(y_labels)
        for i, df in enumerate(dfs):
            if print_results:
                print(df)
            benchmark = benchmarks[i]
            plot_benchmark_result(
                df,
                benchmark.args,
                benchmark.line_names,
                benchmark.x_names,
                y_label,
                benchmark.plot_name,
                show_y_label=i == 0,
                ax=axs[0, i],
            )
        fig.savefig(filepath, bbox_inches="tight")
        plt.clf()
