import os
import random
import argparse

def listdir_fullpath(d, day):
    return [os.path.join(day, os.path.splitext(f)[0]) for f in os.listdir(d) if f.endswith('.mp4')]

parser = argparse.ArgumentParser(description='Sample consultation videos')
parser.add_argument('-d', '--doc',type=str, help="Doctor to be analyzed")
parser.add_argument('-m', '--mode',type=str, help="Presential or Virtual")
args = parser.parse_args()

doctor_dir = args.doc
mode_dir = args.mode
day_lst = os.listdir(f'{doctor_dir}/{mode_dir}')

# Get videos of Presential consultations in a list
lst = []


for day in day_lst:
    lst.extend(listdir_fullpath(f'{doctor_dir}/{mode_dir}/{day}', day))

   
print(lst)
index_lst = random.sample(range(len(lst)), 20) 


with open(f'{doctor_dir}/sampled_{mode_dir}.txt', 'w') as p_f:
    for i in index_lst:
        p_f.write(lst[i]+'\n')
        

