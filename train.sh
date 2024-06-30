#!/usr/bin/env bash

first_letter_upper(){
  str=$1
  first_letter=${str:0:1}
  other_letters=${str:1}
  first_letter=$(echo ${first_letter} | tr '[a-z]' '[A-Z]')
  result=${first_letter}${other_letters}
  echo ${result}
}

players="ace baker casey darling ellis fox"
if [[ $# -gt 0 ]]; then
  players=$*
fi

mkdir -p logs
for who in ${players}; do
  training_set_path="data/train/sgf data/train/histories"
  for d in $training_set_path
  do
    find "$d" -type f -name "*.tar.gz" | while read -r f
    do
      tar -xzf ${f} -C ${d}
    done
  done
  nohup python3 -m app.train $(first_letter_upper ${who}) 1>data/logs/${who}.log 2>&1 &
done