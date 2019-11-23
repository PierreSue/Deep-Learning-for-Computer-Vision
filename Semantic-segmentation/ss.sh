#!/bin/bash

test_dir="$1"
output_dir="$2"

python3 parse_sat.py -t $1

if [ ! -d $2 ]; then
  rm -r $2
  mkdir $2
fi

python3 test.py -m ./model/FCN32s_model_final.h5 -o $2

