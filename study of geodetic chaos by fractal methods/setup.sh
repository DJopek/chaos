#!/bin/zsh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

pip install -r requirements.txt

mkdir -p ./data

cd Gravitacek-2
mkdir -p build
cd build
cmake ..
cmake --build .

cp -r ./* "$SCRIPT_DIR/build/"

cd "$SCRIPT_DIR/build"
