#!/bin/bash

find "${1:-.}" -name "*.pdf" -print0 | xargs -0 -I {} sh -c 'convert -density 600 "$1" "${1%.pdf}.png"' _ {}
