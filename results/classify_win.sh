#!/bin/bash
rm D5/Presential/proc_res/Totals/Stats.csv
for file in D5/Presential/*.csv; do 
	echo "Process $(basename $file)"   
    full=$(basename $file)
    pat='[0-9][0-9]-[0-9][0-9]-[0-9][0-9][0-9][0-9]'
    [[ $(basename $file) =~ $pat ]] # $pat must be unquoted
    date=${BASH_REMATCH[0]}
    suffix="_gaze360_out.csv"
    f=${full%"$suffix"}
    python final_classify.py -i $f -d D5/Presential -c ../Calib/openfiles/$date/Presential/*
done
   
