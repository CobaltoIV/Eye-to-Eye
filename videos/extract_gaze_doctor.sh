#!/bin/bash
while getopts i:e:d: flag
do
    case "${flag}" in
        i) int_params=${OPTARG};;
        e) ext_params=${OPTARG};;
        d) doctor=${OPTARG};;
    esac
done

echo "Intrinsic Parameters file: $int_params"
echo "Extrinsic Parameters file: $ext_params"
echo "Doctor and mode directory: $doctor"



for file in $doctor/*.mp4; do 
    echo $file
    video=$(basename ${file[@]%.mp4})
    echo $video

    cd /home/vislab/opengaze/exe/build
    
    ./bin/GazeVisualization -d -t video -i /home/vislab/Eye-to-Eye/videos/$file --calib_screen /home/vislab/Eye-to-Eye/Calib/openfiles/$ext_params.yml --output /home/vislab/Eye-to-Eye/results/$doctor
    
    echo "OpenGaze Processing Done"
    
    cd /home/vislab/Eye-to-Eye/Gaze_Extraction
    
    source /home/vislab/anaconda3/etc/profile.d/conda.sh
    
    conda activate gaze_env
    
    python3 gaze_processing.py -f /home/vislab/Eye-to-Eye/videos/$file -ip ../Calib/openfiles/$int_params.yml -ep ../Calib/openfiles/$ext_params.yml -d $doctor
    
    conda deactivate
    
    echo "Gaze360 Processing Done"

    cd /home/vislab/Eye-to-Eye/results

    python3 classify_consult.py -i $video -c ../Calib/openfiles/$ext_params.yml -d $doctor

done

