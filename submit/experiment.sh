#!/bin/bash
. ~/venv/bin/activate

python3 condor_experiment.py $(($1+1)) $2 # 0 reserved for combined result

