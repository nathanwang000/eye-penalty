#!/bin/bash
. ~/venv/bin/activate

python3 condor_paramSearch.py $1 $2
