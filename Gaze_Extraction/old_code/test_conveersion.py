import pandas as pd
import datetime
import argparse
import numpy as np
from gazeconversion import Gaze3DTo2D

import cv2

def mapToDisplay(gaze, origin, mon_normal, mon_corners, rmat, tvec, mon_w, mon_h):
    
    gaze_len = mon_normal.dot(mon_corners[0].transpose() - origin)/mon_normal.dot(gaze)
    
    gaze_pos_cam = origin + gaze_len*gaze

    gaze_pos = np.matmul(np.linalg.inv(rmat), gaze_pos_cam-tvec)
    gaze2d = []
    gaze2d.append( gaze_pos[0] / mon_w)
    gaze2d.append( gaze_pos[1] / mon_h)
    
    return gaze2d;

def convert_gaze(row, rmat, tvec, mon_w,mon_h):
    origin =row[["facex", "facey", "facez"]].to_numpy()
    gaze = row[["3d_x", "3d_y", "3d_z"]].to_numpy()
    
    gaze_2d = Gaze3DTo2D(gaze, origin, rmat, tvec)
    return [gaze_2d[0]/mon_w, gaze_2d[1]/mon_h]

def drawGaze(image, gazevec, f_center):
    points = np.array([f_center, f_center + gazevec*300])
    #print(points.shape)
    #print(points)
    end_points, _ = cv2.projectPoints(points,np.zeros((3,1)),np.zeros((3,1)),cam_matrix,dist_coeff)
    start_pt =  (int(end_points[0][0][0]), int(end_points[0][0][1]))
    end_pt  = (int(end_points[1][0][0]), int(end_points[1][0][1]))
    
    cv2.line(image, start_pt, end_pt, (255,255,255), 2)
    cv2.circle(image, start_pt, 5, (255,255,255),-1)
    cv2.circle(image, end_pt,5, (255,255,255), -1)
    return image

def main(args):
        # Loading Screen Size
    fs = cv2.FileStorage(
       args.config, cv2.FILE_STORAGE_READ)
    mon_w = fs.getNode("monitor_W").real()
    mon_h = fs.getNode("monitor_H").real()
    ext_rmat = fs.getNode("monitor_R").mat()
    ext_tvec = fs.getNode("monitor_T").mat()
        
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

    if not args.output_file:
        args.output_file = 'proc_res/'+args.input_file.replace('.csv', '_res.csv')
        print('Output file not defined \n Saving results to ' + args.output_file)
        
    # Define Collumns of .csv
    nm = ["frame", "f_found", "f_confidence", "facex", "facey", "facez",
        "2d_x", "2d_y", "3d_x", "3d_y", "3d_z"]

    # Video fps
    fps = 15

    # Load samples from .csv
    df = pd.read_csv(args.input_file, names=nm)

    
    df['converted'] = df.apply(convert_gaze, args=(ext_rmat, ext_tvec,mon_w, mon_h), axis=1)
    
    df.to_csv("res.csv")
    
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process gaze output csv file')
    
    parser.add_argument('-i', '--input_file',type=str, help="Input csv location")
    parser.add_argument('-o', '--output_file',type=str, help="Output csv location")
    parser.add_argument('-c', '--config', type=str , help='Configuration file', default="/home/vislab/RecordingSetup/Calib/openfiles/my_pc.yml")
    
    args = parser.parse_args()
    main(args)
    