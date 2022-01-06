# coding: utf-8

__author__ = 'cleardusk'

import argparse
import imageio
from tqdm import tqdm
import yaml
import numpy as np
import pandas as pd
import csv

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


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

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
    reader = imageio.get_reader(args.video_fp)

    fps = reader.get_meta_data()['fps']

    suffix = get_suffix(args.video_fp)
    csv_outfile = args.video_fp.replace(suffix, "_land.csv")
    '''    
    video_wfp = f'examples/results/videos/{fn.replace(suffix, "")}_{args.opt}.mp4'
    writer = imageio.get_writer(video_wfp, fps=fps)
    '''
    full = []
    ver = [[0],[0],[0]]
    # run
    dense_flag = args.opt in (
        '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
    pre_ver = None
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
            pre_ver = ver  # for tracking
            
            '''
            if args.opt == '2d_sparse':
                temp = cv_draw_landmark(frame_bgr, ver)
                res = viz_pose(temp, param_lst, [ver])
            else:
                raise ValueError(f'Unknown opt {args.opt}')
            
            writer.append_data(res[..., ::-1])  # BGR->RGB
            
        else:
            writer.append_data(frame)  # BGR->RGB
        '''
        full.append([ver,boxes])


    
    with open(csv_outfile, 'w') as f:
        csvwriter = csv.writer(f)
        for i in full:
            csvwriter.writerow(i[0][0])
            csvwriter.writerow(i[0][1])
            if i[1]:
                csvwriter.writerows(i[1])
            else:
                csvwriter.writerow(['Face Not Found'])
    print("Facial Landmarking Done")
    '''
    writer.close()
    print(f'Dump to {video_wfp}')
    '''


if __name__ == '__main__':
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

    args = parser.parse_args()
    main(args)
