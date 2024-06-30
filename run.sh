#!/usr/bin/env bash

mkdir -p data/train/histories
mkdir -p data/train/sgf
mkdir -p data/logs

python3 -m app.run $*