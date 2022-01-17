#!/bin/bash
while getopts d: flag
do
    case "${flag}" in
        d) directory=${OPTARG};
    esac
done
echo "Processing data for directory $directory"  

cd Gaze_Extraction

for mode in $directory/*; do
    if [ -d $mode -a "$(ls -A $mode )" ]
    then
    echo "Processing $(basename $mode) consultations" 
    for file in $mode/*.mp4; do 
        echo "Process $(basename $file)"   
        python3 video_landmarks.py -f "$file"
        echo "Landmarking done for $file"  
    done
    fi
done


