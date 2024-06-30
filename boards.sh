#!/usr/bin/env bash

who=ace
if [[ -n "$1" ]];then
  who=$1
fi

grep -E "(^=|^ |Game|player)" data/logs/${who}.log
