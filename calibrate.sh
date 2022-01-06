# Prepare data for extrinsic calibration
cd Calib/Preprocess
python undistort.py
python find_corners.py
cp input* ../data

# Generate the correct model for the Extrinsic Calibration
cd ../data
python gen_model.py

cd ..
cp data/*.jpg data/*.txt /home/vislab/takahashi2012cvpr/data

#Perform Extrinsic Calibration routine
cd ..
sudo /usr/local/MATLAB/R2018a/bin/matlab -sd "/home/vislab/takahashi2012cvpr/matlab" -r "demo" 


echo "Finished"
#Put in Git
#git add .
#git commit -m "Add Extrinsic Parameters"
#git push
