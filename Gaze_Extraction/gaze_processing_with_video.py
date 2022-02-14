# coding: utf-8
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
from utils.functions import cv_draw_landmark, get_suffix
from utils.tddfa_util import str2bool
from gaze360.code.model import GazeLSTM

WIDTH = 1280
HEIGHT = 720
def drawGaze(image, gazevec, f_center,color, cam_matrix, dist_coeff):
    points = np.array([f_center, f_center + gazevec*300])
    end_points, _ = cv2.projectPoints(points,np.zeros((3,1)),np.zeros((3,1)),cam_matrix,dist_coeff)
    start_pt =  (int(end_points[0][0][0]), int(end_points[0][0][1]))
    end_pt  = (int(end_points[1][0][0]), int(end_points[1][0][1]))
    
    cv2.line(image, start_pt, end_pt, color, 2)
    cv2.circle(image, start_pt, 5, color,-1)
    cv2.circle(image, end_pt,5, color, -1)
    return image


def annotateFrame(image, mpii_gaze, gaze360_gaze, avg_gaze, f_center, origin, bbox, cam_matrix, dist_coeff):
    image = drawGaze(image,gaze360_gaze,f_center,(255,255,255), cam_matrix, dist_coeff)
    image = drawGaze(image, mpii_gaze, origin, (255,0,0), cam_matrix, dist_coeff)
    image = drawGaze(image,avg_gaze,f_center,(0,255,0), cam_matrix, dist_coeff)
    image = cv2.rectangle(image, (bbox[0]-50,bbox[1]-50), (bbox[2]+50,bbox[3]+50), (255,255,255))
    image = image.astype(np.uint8)
    return image

def spherical2cartesial(x): 
    output = torch.zeros(x.size(0),3) 
    output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])
    output[:,2] = torch.cos(x[:,1])*torch.cos(x[:,0])
    return output


def estimateHeadPose(landmarks,cam_matrix, face_model):
    
    cmatrix = np.float64(cam_matrix)
    zero_dist = np.float64(np.zeros((5,1)))
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, cmatrix,  zero_dist)
    
    rmat, _ = cv2.Rodrigues(rvec)
    
    Fc = np.matmul(rmat, face_model.transpose()) + tvec
    
    face_center = np.sum(Fc,axis=1)/6

    return face_center, rmat, tvec

def readIntParams(f_int_params):
    int_fs = cv2.FileStorage(f_int_params, cv2.FILE_STORAGE_READ)
    cam_matrix = int_fs.getNode("camera_matrix").mat()
    dist_coeff = int_fs.getNode("dist_coeffs").mat()
    return cam_matrix, dist_coeff

def readExtParams(f_ext_params):
    ext_fs = cv2.FileStorage(args.ext_params, cv2.FILE_STORAGE_READ)
    mon_w= ext_fs.getNode("monitor_W").real()
    mon_h = ext_fs.getNode("monitor_H").real()
    ext_rmat = ext_fs.getNode("monitor_R").mat()
    ext_tvec = ext_fs.getNode("monitor_T").mat()
    return mon_w,mon_h, ext_rmat, ext_tvec

def convertToScreenCoordinates(point_of_regard, mon_w, mon_h):
    return [point_of_regard[0]/mon_w, point_of_regard[1]/mon_h]
    
def getIm(frame, ):
    image = frame.copy()  
    image = cv2.resize(image,(WIDTH,HEIGHT))
    image = image.astype(float)
    return image

def main(args):
    

    image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    nm = ["frame", "f_found", "f_confidence", "facex", "facey", "facez",
            "2d_x", "2d_y", "3d_x", "3d_y", "3d_z"]


    cam_matrix, dist_coeff = readIntParams(args.int_params)


    mon_w, mon_h ,ext_rmat,ext_tvec = readExtParams(args.ext_params)


    face_fs = cv2.FileStorage("face_model.yml", cv2.FILE_STORAGE_READ)
    face_model = face_fs.getNode("face_model").mat().transpose()


    landmark_indices= [36, 39, 42, 45, 48, 54]



    # Given a video path
    fn = args.video_fp.split('/')[-1]
    gaze_mpii = fn.replace('.mp4', '')

    reader = imageio.get_reader(args.video_fp) 

    suffix = get_suffix(args.video_fp)

    fps = reader.get_meta_data()['fps']
    annotated_video = args.video_fp.replace(suffix, "_annotated.mp4")
    out = imageio.get_writer(annotated_video,fps=fps)

    csv_outfile = f'../results/{args.doctor}/{fn.replace(suffix, "_gaze360_out.csv")}'

    avg_csv_outfile = f'../results/{args.doctor}/{fn.replace(suffix, "_avg_gaze360_out.csv")}'

    land_file = args.video_fp.replace(suffix, "_land.csv")

    W = max(int(fps//8),1)
    full = []
    ver = [[0],[0],[0]]
    frames_with_people = dict()
    image_lst = []

    # Read Facial Landmark Detection
    df = pd.read_csv(land_file, dtype=str, names=range(0,68))

    frames_with_people = dict()
    frames = int(df.shape[0]/3)
    for i in range(0, frames):
        ver0 = np.float64(np.asarray(df.loc[i*3]))
        ver1 = np.float64(np.asarray(df.loc[i*3 + 1]))
        if df.loc[i*3 + 2][0] == "Face Not Found":
            box= np.zeros(5)
        else:
            box = np.float64(np.asarray(df.loc[i*3 + 2]))[0:5]
        #If Face is found
        if box[0] != 0:
            landmarks = np.asarray([ver0, ver1])
            l = landmarks[:,landmark_indices]
            face_center, head_r, head_t = estimateHeadPose(l.transpose(),cam_matrix, face_model)
            frames_with_people[i] = [-face_center, box, head_r, head_t] 
        
        
        
    print('Read Landmarks from ' + land_file)
    res = []
    avg_res = []
    ind_frames = frames_with_people.keys()

        
    #Run Gaze Detection

    mpii_df = pd.read_csv(f'../results/{args.doctor}/{gaze_mpii}_gaze_output.csv', names=nm)

    #Load Gaze360 model
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
            #With the 3 next images saved we can proceed to use Gaze360
            i =  n-4
            if i < 4:
                image_lst.append(frame)
                image = image_lst[i]
            else:
                image_lst.pop(0)
                image_lst.append(frame)
                image = image_lst[3]
            
        
            if i not in ind_frames:  # If face was not detected in frame
                res.append([i,0,0,np.zeros((3,1)),np.zeros((2,1)), np.zeros((3,1))])
                avg_res.append([i,0,0,np.zeros((3,1)),np.zeros((2,1)), np.zeros((3,1))])
                out.append_data(image)
            else:
                box = frames_with_people[i][1]
                face_center = frames_with_people[i][0]
                head_r = frames_with_people[i][2]
                input_image = torch.zeros(7,3,224,224)
                count = 0
                # Use cancatenae the seven frames with ith frame in the middle
                for j in range(i-3*W,i+4*W,W):         
                    if j in ind_frames:
                        jbox = frames_with_people[j][1]
                        new_im = Image.fromarray(image_lst[min(j,count)],'RGB')    
                        new_im = new_im.crop((jbox[0]-50,jbox[1]-50,jbox[2]+50,jbox[3]+50))
                        input_image[count,:,:,:] = image_normalize(transforms.ToTensor()(transforms.Resize((224,224))(new_im)))
                    else:
                        new_im = Image.fromarray(image,'RGB')    
                        new_im = new_im.crop((box[0]-50,box[1]-50,box[2]+50,box[3]+50))
                        input_image[count,:,:,:] = image_normalize(transforms.ToTensor()(transforms.Resize((224,224))(new_im)))
                    count = count+1
                #Convert bounding box pixels to int
                bbox = np.asarray(box).astype(int)   
                
                origin = mpii_df.iloc[i][["facex", "facey", "facez"]].to_numpy()
                mpiigaze = mpii_df.iloc[i][["3d_x", "3d_y", "3d_z"]].to_numpy()
                #Get 3d gaze coords
                output_gaze,_ = model(input_image.view(1,7,3,224,224).cuda())
                gaze = spherical2cartesial(output_gaze).detach().numpy()
                gaze = -gaze.reshape((-1))
                
                #Average outputs of Mpii and Gaze360
                avg_gaze = np.add(mpiigaze,gaze)/2
                
                #Convert to 2D
                avg_gaze2d = Gaze3DTo2D(avg_gaze,face_center,ext_rmat,ext_tvec)
                avg_gaze2d = convertToScreenCoordinates(avg_gaze2d, mon_w, mon_h)
                
                gaze2d = Gaze3DTo2D(gaze,face_center,ext_rmat,ext_tvec)
                gaze2d = convertToScreenCoordinates(gaze2d, mon_w, mon_h)
                
                res.append([i, 1, box[4], face_center[0], face_center[1], face_center[2], gaze2d[0][0], gaze2d[1][0],gaze[0], gaze[1], gaze[2]])
                avg_res.append([i, 1, box[4], face_center[0], face_center[1], face_center[2], avg_gaze2d[0][0], avg_gaze2d[1][0], avg_gaze[0], avg_gaze[1], avg_gaze[2]])
                
                video_image = getIm(image)
                
                #Draw gaze and face rectangle in image 
                try:
                    video_image = annotateFrame(video_image,mpiigaze,gaze, avg_gaze, face_center, origin, bbox, cam_matrix, dist_coeff)
                    out.append_data(video_image)
                except:
                    out.append_data(video_image)
                   
		           
                
                
    print('Dump annotated video to ' + annotated_video)
    out.close()
    print('Gaze Extraction done \n saving results to '+ csv_outfile + ' and ' + avg_csv_outfile)
    res_df = pd.DataFrame(data = res, columns = ["frame", "f_found", "f_confidence", "face_centerx", "face_centery", "face_centerz",
            "2d_x", "2d_y", "3d_x", "3d_y", "3d_z"]
    )
    res_df.to_csv(csv_outfile)

    avg_res_df = pd.DataFrame(data = avg_res, columns = ["frame", "f_found", "f_confidence", "face_centerx", "face_centery", "face_centerz",
            "2d_x", "2d_y", "3d_x", "3d_y", "3d_z"]
    )
    avg_res_df.to_csv(avg_csv_outfile)

if __name__ == '__main__':      
    parser = argparse.ArgumentParser(
        description='The demo of video of Eye-To-Eye')

    parser.add_argument('-f', '--video_fp', type=str, required=True)
    parser.add_argument('-d', '--doctor', type=str, required=True)
    parser.add_argument('-ip', '--int_params', type=str, help = "Intrinsic Parameters of camera")
    parser.add_argument('-ep', '--ext_params', type=str, help = "Extrinsic Parameters of camera")



    args = parser.parse_args()
    main(args)
