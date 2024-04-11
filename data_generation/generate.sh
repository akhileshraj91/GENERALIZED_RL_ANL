#!/bin/bash


export LC_ALL=C  # ensure we are working with known locales
#set -e -u -f -o pipefail # safer shell script

declare -r PROGRAM=${0##*/}

python identification.py ./new_stairs.yaml -- ones-stream-full 33554432 10000
python identification.py ./new_stairs.yaml -- ones-stream-add 33554432 10000
python identification.py ./new_stairs.yaml -- ones-stream-triad 33554432 10000
python identification.py ./new_stairs.yaml -- ones-stream-scale 33554432 10000
python identification.py ./new_stairs.yaml -- ones-stream-copy 33554432 10000
python identification.py ./new_stairs.yaml -- ones-npb-ep 22 10000
python identification.py ./new_stairs.yaml -- ones-solvers-cg 10000 poor 0 10000
python identification.py ./new_stairs.yaml -- ones-solvers-bicgstab 20000 poor 0 10000



