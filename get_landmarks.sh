#!/bin/bash
cd Recording
for dir in */; do
    echo "Processing data for doctor $(basename $dir)"
    for day in $dir*/; do
        echo "Processing Day $(basename $day)"
        for mode in $day*/; do
            echo "Processing $(basename $mode) consultations"    
            for file in $mode*.mp4; do 
            echo "Process $(basename $file)"   
            cd ~/3DDFA_V2
            python3 video_landmarks.py -f ~/RecordingSetup/Recording/"$file"
            cd ~/RecordingSetup/Recording
            echo "Landmarking done for $file"     
            done
        done
    done
    cp -R $dir ../videos/
    echo "$(basename $dir) recordings copied to videos directory"
done
