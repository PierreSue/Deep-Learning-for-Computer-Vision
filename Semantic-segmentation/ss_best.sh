#!/bin/bash

test_dir="$1"
output_dir="$2"

python3 parse_sat.py -t test_dir

if [ ! -d output_dir ]; then
  rm -r output_dir
  mkdir output_dir
fi

python3 test.py -m ./model/FCN8s_model_final.h5 -o output_dir

