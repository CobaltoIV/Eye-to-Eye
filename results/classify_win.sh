#!/bin/bash
rm D6/Virtual/proc_res/Totals/Stats.csv
for file in D6/Virtual/*.csv; do 
	echo "Process $(basename $file)"   
    full=$(basename $file)
    pat='[0-9][0-9]-[0-9][0-9]-[0-9][0-9][0-9][0-9]'
    [[ $(basename $file) =~ $pat ]] # $pat must be unquoted
    date=${BASH_REMATCH[0]}
    suffix="_gaze360_out.csv"
    f=${full%"$suffix"}
    python final_classify.py -i $f -d D6/Virtual -c ../Calib/openfiles/$date/Virtual/*
done
   
