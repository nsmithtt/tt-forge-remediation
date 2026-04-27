#!/bin/bash

TT_FORGE_MODELS_BRANCH="${1:-}"

git submodule update --init --recursive

if [ -n "$TT_FORGE_MODELS_BRANCH" ]; then
    cd tt-xla/third_party/tt_forge_models
    git fetch origin
    git checkout "origin/$TT_FORGE_MODELS_BRANCH" --detach
    cd ../../..
fi

cd tt-xla
source venv/activate
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DTTMLIR_SOURCE_DIR_OVERRIDE=../tt-mlir -DTTMLIR_TTMETAL_SOURCE_DIR=../tt-metal
