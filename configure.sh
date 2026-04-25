#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <tt-forge-models-branch>"
    exit 1
fi

git submodule update --init --recursive

cd tt-xla

pushd third_party/tt_forge_modules
git checkout $1
popd

source venv/activate
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DTTMLIR_SOURCE_DIR_OVERRIDE=../tt-mlir -DTTMLIR_TTMETAL_SOURCE_DIR=../tt-metal
