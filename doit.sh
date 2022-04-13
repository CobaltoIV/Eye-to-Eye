bash day_land.sh

cp /media/vislab/My_Passport/Tese/videos/Neurologia/D8/Virtual/06-04-2022/* /home/vislab/Eye-to-Eye/videos/D8/Virtual
cp /media/vislab/My_Passport/Tese/videos/Endocrinologia/D10/Presential/25-03-2022/* /home/vislab/Eye-to-Eye/videos/D10/Presential

cd videos/Testing 
bash test_videos.sh
cd ..
bash process_multiple.sh