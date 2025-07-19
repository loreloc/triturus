#!/bin/bash

SCRIPTS_PATH="scripts"
GIT_REV="$(git rev-parse HEAD)"
GIT_SHORT_REV="$(git rev-parse --short HEAD)"
RESULTS_FILEPATH="benchmarks-${GIT_SHORT_REV}.txt"

echo "Running benchmarks and storing results to ${RESULTS_FILEPATH} ..."
echo -e "Git Rev: ${GIT_REV}\n" > "${RESULTS_FILEPATH}"
"${SCRIPTS_PATH}/run_benchmarks.sh" >> "${RESULTS_FILEPATH}"
