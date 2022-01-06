# coding: utf-8

__author__ = 'cleardusk'

import argparse
import imageio
import sys

from gazeconversion import Gaze3DTo2D
sys.path.append('gaze360/code/')

import json
from tqdm import tqdm
import yaml
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import csv
import random
from PIL import Image
import math
import ctypes.util
from numpy.linalg import inv

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
# from utils.render_ctypes import render
from utils.functions import cv_draw_landmark, get_suffix
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.tddfa_util import str2bool
from gaze360.code.model import GazeLSTM

def drawGaze(image, gazevec, f_center,color):
    points = np.array([f_center, f_center + gazevec*300])
    end_points, _ = cv2.projectPoints(points,np.zeros((3,1)),np.zeros((3,1)),cam_matrix,dist_coeff)
    start_pt =  (int(end_points[0][0][0]), int(end_points[0][0][1]))
    end_pt  = (int(end_points[1][0][0]), int(end_points[1][0][1]))
    
    cv2.line(image, start_pt, end_pt, color, 2)
    cv2.circle(image, start_pt, 5, color,-1)
    cv2.circle(image, end_pt,5, color, -1)
    return image

    
def spherical2cartesial(x): 
    output = torch.zeros(x.size(0),3) 
    output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])
    output[:,2] = torch.cos(x[:,1])*torch.cos(x[:,0])
    return output


def estimateHeadPose(landmarks):
    
    cmatrix = np.float64(cam_matrix)
    zero_dist = np.float64(np.zeros((5,1)))
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, cmatrix,  zero_dist)
    
    rmat, _ = cv2.Rodrigues(rvec)
    
    Fc = np.matmul(rmat, face_model.transpose()) + tvec
    
    face_center = np.sum(Fc,axis=1)/6

    return face_center, rmat, tvec


def getIm(frame):
    image = frame.copy()  
    image = cv2.resize(image,(WIDTH,HEIGHT))
    image = image.astype(float)
    return image

parser = argparse.ArgumentParser(
    description='The demo of video of 3DDFA_V2')
parser.add_argument('-c', '--config', type=str,
                    default='configs/mb1_120x120.yml')
parser.add_argument('-f', '--video_fp', type=str)
parser.add_argument('-m', '--mode', default='cpu',
                    type=str, help='gpu or cpu mode')
parser.add_argument('--show_flag', type=str2bool, default='true',
                    help='whether to show the visualization result')

parser.add_argument('-o', '--opt', type=str, default='2d_sparse', choices=[
                    '2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
parser.add_argument('--onnx', action='store_true', default=False)
parser.add_argument('-ip', '--int_params', type=str, help = "Intrinsic Parameters of camera")
parser.add_argument('-ep', '--ext_params', type=str, help = "Exxtrinsic Parameters of camera")


args = parser.parse_args()
cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


nm = ["frame", "f_found", "f_confidence", "facex", "facey", "facez",
        "2d_x", "2d_y", "3d_x", "3d_y", "3d_z"]


int_fs = cv2.FileStorage(args.int_params, cv2.FILE_STORAGE_READ)
cam_matrix = int_fs.getNode("camera_matrix").mat()
dist_coeff = int_fs.getNode("dist_coeffs").mat()


ext_fs = cv2.FileStorage(args.ext_params, cv2.FILE_STORAGE_READ)
mon_w= ext_fs.getNode("monitor_W").real()
mon_h = ext_fs.getNode("monitor_H").real()
ext_rmat = ext_fs.getNode("monitor_R").mat()
ext_tvec = ext_fs.getNode("monitor_T").mat()


corners = np.array([
                    (0,0,0),
                    (mon_w,0,0),
                    (0,mon_h,0),
                    (mon_w, mon_h,0)
                ])
mon_corners = []
for c in corners:
    mon_corners.append(np.matmul(ext_rmat,c) + ext_tvec.transpose())


normal = np.array([0,0,1], dtype="float") 
mon_normal = np.matmul(ext_rmat, normal)
mon_normal = np.asarray(mon_normal, dtype="float")

#mapToDisplay(np.array([12,14,15]), np.array([13,78,65]))
face_fs = cv2.FileStorage("face_model.yml", cv2.FILE_STORAGE_READ)
face_model = face_fs.getNode("face_model").mat().transpose()


landmark_indices= [36, 39, 42, 45, 48, 54]


# Init FaceBoxes and TDDFA, recommend using onnx flag
if args.onnx:
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '8'

    from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
    from TDDFA_ONNX import TDDFA_ONNX

    face_boxes = FaceBoxes_ONNX()
    tddfa = TDDFA_ONNX(**cfg)
else:
    gpu_mode = args.mode == 'gpu'
    tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
    face_boxes = FaceBoxes()

# Given a video path
fn = args.video_fp.split('/')[-1]
gaze_mpii = fn.replace('.mp4', '')
reader = imageio.get_reader(args.video_fp) 
suffix = get_suffix(args.video_fp)

df = pd.read_csv(f'/home/vislab/RecordingSetup/results/{gaze_mpii}_gaze_output.csv', names=nm)

fps = reader.get_meta_data()['fps']

out = imageio.get_writer(args.video_fp.replace(suffix, "_annotated.mp4"),fps=fps)

csv_outfile = f'/home/vislab/RecordingSetup/results/{fn.replace(suffix, "_gaze360_out.csv")}'
avg_csv_outfile = f'/home/vislab/RecordingSetup/results/{fn.replace(suffix, "avg_gaze360_out.csv")}'

W = max(int(fps//8),1)
WIDTH = 1280
HEIGHT = 720
full = []
ver = [[0],[0],[0]]
frames_with_people = dict()
image_lst = []
# run Facial Landmark Detection
dense_flag = args.opt in (
    '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
for i, frame in tqdm(enumerate(reader)):
    frame_bgr = frame[..., ::-1]  # RGB->BGR        
    boxes = face_boxes(frame_bgr)
    if boxes:
        boxes = [boxes[0]]
        param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
        ver = tddfa.recon_vers(
            param_lst, roi_box_lst, dense_flag=dense_flag)[0]
        # refine
        param_lst, roi_box_lst = tddfa(
            frame_bgr, [ver], crop_policy='landmark')
        ver = tddfa.recon_vers(
            param_lst, roi_box_lst, dense_flag=dense_flag)[0]
        # TODO alter to create frames with people list
        landmarks = np.asarray([np.asarray(ver[0]), np.asarray(ver[1])])
        l = landmarks[:,landmark_indices]
        face_center, head_r, head_t = estimateHeadPose(l.transpose())  
        frames_with_people[i] = [-face_center,boxes, head_r, head_t] 
print('Facial Landmarking done')
res = []
avg_res = []
ind_frames = frames_with_people.keys()

#Run Gaze Detection

model = GazeLSTM()
model = torch.nn.DataParallel(model).cuda()
model.cuda()
checkpoint = torch.load('gaze360_model.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

for n, frame in tqdm(enumerate(reader)): 
    if n < 4:
        image_lst.append(frame)
    else:
        i =  n-4
        if i < 4:
            image_lst.append(frame)
            image = image_lst[i]
        else:
            image_lst.pop(0)
            image_lst.append(frame)
            image = image_lst[3]
        
        
        if i not in ind_frames:
            res.append([i,0,0,np.zeros((3,1)),np.zeros((2,1)), np.zeros((3,1))])
            out.append_data(image)
        else:
            box = frames_with_people[i][1][0]
            face_center = frames_with_people[i][0]
            head_r = frames_with_people[i][2]
            input_image = torch.zeros(7,3,224,224)
            count = 0
            for j in range(i-3*W,i+4*W,W):         
                if j in ind_frames:
                    jbox = frames_with_people[j][1][0]
                    new_im = Image.fromarray(image_lst[min(j,count)],'RGB')    
                    new_im = new_im.crop((jbox[0]-50,jbox[1]-50,jbox[2]+50,jbox[3]+50))
                    input_image[count,:,:,:] = image_normalize(transforms.ToTensor()(transforms.Resize((224,224))(new_im)))
                else:
                    new_im = Image.fromarray(image,'RGB')    
                    new_im = new_im.crop((box[0]-50,box[1]-50,box[2]+50,box[3]+50))
                    input_image[count,:,:,:] = image_normalize(transforms.ToTensor()(transforms.Resize((224,224))(new_im)))
                count = count+1
                
            bbox = np.asarray(box).astype(int)   
            
            origin = df.iloc[i][["facex", "facey", "facez"]].to_numpy()
            mpiigaze =df.iloc[i][["3d_x", "3d_y", "3d_z"]].to_numpy()
            #Get 3d gaze coords
            output_gaze,_ = model(input_image.view(1,7,3,224,224).cuda())
            gaze = spherical2cartesial(output_gaze).detach().numpy()
            gaze = -gaze.reshape((-1))
            
            avg_gaze = np.add(mpiigaze,gaze)/2
            avg_gaze2d = Gaze3DTo2D(avg_gaze,face_center,ext_rmat,ext_tvec)
            avg_gaze2d[0] = avg_gaze2d[0]/mon_w
            avg_gaze2d[1] = avg_gaze2d[1]/mon_h
            
            #Convert to 2D
            gaze2d = Gaze3DTo2D(gaze,face_center,ext_rmat,ext_tvec)
            gaze2d[0] = gaze2d[0]/mon_w
            gaze2d[1] = gaze2d[1]/mon_h
            
            res.append([i, 1, box[4], face_center[0], face_center[1], face_center[2], gaze2d[0][0], gaze2d[1][0],gaze[0], gaze[1], gaze[2]])
            avg_res.append([i, 1, box[4], face_center[0], face_center[1], face_center[2], avg_gaze2d[0][0], avg_gaze2d[1][0], avg_gaze[0], avg_gaze[1], avg_gaze[2]])
            image = getIm(image)
            
            #Draw gaze and face rectangle in image 
            image = drawGaze(image,gaze,face_center,(255,255,255))
            image = drawGaze(image, mpiigaze, origin, (255,0,0))
            image = drawGaze(image,avg_gaze,face_center,(0,255,0))
            image = cv2.rectangle(image, (bbox[0]-50,bbox[1]-50), (bbox[2]+50,bbox[3]+50), (255,255,255))
            image = image.astype(np.uint8)
            out.append_data(image)
out.close()
print('Gaze Extraction done \n saving results to '+ csv_outfile)
res_df = pd.DataFrame(data = res, columns = ["frame", "f_found", "f_confidence", "face_centerx", "face_centery", "face_centerz",
        "2d_x", "2d_y", "3d_x", "3d_y", "3d_z"]
)
res_df.to_csv(csv_outfile)

avg_res_df = pd.DataFrame(data = avg_res, columns = ["frame", "f_found", "f_confidence", "face_centerx", "face_centery", "face_centerz",
        "2d_x", "2d_y", "3d_x", "3d_y", "3d_z"]
)
avg_res_df.to_csv(avg_csv_outfile)



    


