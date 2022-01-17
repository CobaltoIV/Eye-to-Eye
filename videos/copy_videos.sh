#!/bin/bash
while getopts m:d: flag
do
    case "${flag}" in
        m) mode=${OPTARG};;
        d) doctor=${OPTARG};;
    esac
done
echo "Copying chose videos from $doctor/$mode"

while read p; do
  echo "$p"
  cp $doctor/$mode/$p.mp4 $doctor/$mode/"$p"_land.csv ~/lol
done < $doctor/sampled_$mode.txt
