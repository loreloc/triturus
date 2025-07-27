#!/bin/bash

export PYTHONPATH=.

BENCHMARKS_PATH="benchmarks"
BENCHMARK_SUITES="add max mm lm2exp lt2exp"

for suite in $BENCHMARK_SUITES
do
    echo "Running benchmark suite '${suite}' ..."
    python "${BENCHMARKS_PATH}/${suite}.py"
    echo
done
