import pandas as pd
import datetime
import argparse
import math
from pandas.core import frame
import cv2

class bounds:
    def __init__(self, x0,x1,y0,y1):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

def get_result(x, y, confidence, boundaries):
    """Classify the gaze sample

    Args:
        x (float): x coodinate on the screen plane
        y (float): y coordinate on the screen plane
        confidence (float): Face detection confidence of OpenFace

    Returns:
        res: Classificcation of sample
    """
    x0 = boundaries.x0
    x1 = boundaries.x1
    y0 = boundaries.y0
    y1 = boundaries.y1
    
    if confidence <= 0.80:  # OpenFace couldn't detect a face
        return "not_in_frame"
    elif x > x1:
        return "Right Of Screen"
    elif x < x0:
        return "Left Of Screen"
    elif x < x1 and x > x0 and y > y1:
        return "Keyboard"
    elif x < x1 and x > x0 and y < y0:
        return "Above Screen"
    else:
        return "Screen"



def get_timestamp(frame, fps):
    """Calculate timestamp in the video

    Args:
        frame (int): Frame number
        fps (int): Video fps

    Returns:
        timestamp: The timestamp in the video for the respective frame
    """
    td = datetime.timedelta(seconds=frame*(1/fps))
    time = "{hour:d}:{min:d}:{sec:.2f}"
    hours = td.seconds//3600
    minutes = td.seconds//60
    second = td.seconds + td.microseconds*10**-6 - minutes*60

    return time.format(hour=hours, min=minutes, sec=second)


def get_coord_screen(res, x, y, monitor_W, monitor_H):
    """Get coordinates in milimiters on the screen

    Args:
        res (string): Classificatio of sample
        x (float): x coordinate on the screen 
        y (float): y coordinate on the screen 
        monitor_W (float): Screen width in millimeters
        monitor_H (float): Screen heigth in millimeters

    """
    if res == "Screen":
        return [x*monitor_W, y*monitor_H]
    else:
        return ["not_valid", "not_valid"]


def add_columns(row, monitor_W, monitor_H, fps):
    """Function applied to data to classify and process all samples
    """
    gaze360_x = row["gaze360_x"]
    gaze360_y = row["gaze360_y"]
    
    mpii_x = row["mpii_x"]
    mpii_y = row["mpii_y"]
    
    avg_x = row["avg_x"]
    avg_y = row["avg_y"]
    
    confidence = row["f_confidence"]
    frame = row["frame"]
    
    mpii_bounds = bounds(0,1,0,1)
    gaze360_bounds = bounds(0.2,1.4,-0.5,1.4)
    
    dist = math.dist([gaze360_x, gaze360_y], [mpii_x, mpii_y])
    
    time = get_timestamp(frame, fps)
    
    gaze360_res = get_result(gaze360_x, gaze360_y, confidence, gaze360_bounds)
    
    mpii_res = get_result(mpii_x, mpii_y, confidence, mpii_bounds)
    
    avg_res = get_result(avg_x, avg_y, confidence, gaze360_bounds)
    gaze360_2d = [gaze360_x*monitor_W, gaze360_y*monitor_H]
    mpii_2d = [mpii_x*monitor_W, mpii_y*monitor_H]

    dist = math.dist(gaze360_2d, mpii_2d)
    diff_3d = [row['gaze360_3d_x']- row['mpii_3d_x'], row['gaze360_3d_y']- row['mpii_3d_y'], row['gaze360_3d_z']- row['mpii_3d_z']]

    return time, gaze360_res, mpii_res, avg_res, dist , gaze360_2d[0], gaze360_2d[1], mpii_2d[0], mpii_2d[1], diff_3d[0], diff_3d[1], diff_3d[2]


def main(args):
    # Loading Screen Size
    if not args.config:
        raise Exception('Please specify configuration file.')
    fs = cv2.FileStorage(
       args.config, cv2.FILE_STORAGE_READ)
    W = fs.getNode("monitor_W").real()
    H = fs.getNode("monitor_H").real()
    
    name = args.input_file
    
    if not args.output_file:
        args.output_file = f'proc_res/{name}_combined_gaze_output.csv'
        print('Output file ot defined saving results to ' + args.output_file)
        
    
    # Video fps
    fps = 15
    print(name)
    nm = ["frame", "f_found", "f_confidence", "facex", "facey", "facez",
        "2d_x", "2d_y", "3d_x", "3d_y", "3d_z"]
    # Load samples from .csv
    gaze_360_out = f'{name}_gaze360_out.csv'
    mpii_out = f'{name}_gaze_output.csv'
    avg_out = f'{name}_avg_gaze360_out.csv'

    
    df_360 = pd.read_csv(gaze_360_out,index_col=0)
   
    df_mpii = pd.read_csv(mpii_out, names=nm)
   
    df_avg = pd.read_csv(avg_out, index_col=0)


    df_combined = pd.DataFrame()

    df_combined['frame'] = df_mpii['frame']
    df_combined['f_confidence'] = df_mpii['f_confidence']
    df_combined['gaze360_x'] = df_360['2d_x']
    df_combined['gaze360_y'] = df_360['2d_y']
    df_combined['mpii_x']  =df_mpii['2d_x']
    df_combined['mpii_y']  =df_mpii['2d_y']
    df_combined['avg_x'] = df_avg['2d_x']
    df_combined['avg_y'] = df_avg['2d_y']
    df_combined['gaze360_3d_x'] = df_360['3d_x']
    df_combined['gaze360_3d_y'] = df_360['3d_y']
    df_combined['gaze360_3d_z'] = df_360['3d_z']
    df_combined['mpii_3d_x']  =df_mpii['3d_x']
    df_combined['mpii_3d_y']  =df_mpii['3d_y']
    df_combined['mpii_3d_z']  =df_mpii['3d_z']


    df_combined[['timestamp', 'gaze360_res', 'mpii_res', 'avg_res', 'dist', '2d_gaze360_x', '2d_gaze360_y', '2d_mpii_x', '2d_mpii_y', 'diff_x', 'diff_y', 'diff_z']] =df_combined.apply(add_columns,args=(W, H, fps), axis=1, result_type='expand')
    
    df_combined[['timestamp', 'gaze360_res', 'mpii_res', 'avg_res', 'dist', 'gaze360_x', 'gaze360_y','mpii_x', 'mpii_y', 'avg_x', 'avg_y']].to_csv(args.output_file)
    
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process gaze output csv file')
    
    parser.add_argument('-i', '--input_file',type=str, help="Input .mp4 name")
    parser.add_argument('-o', '--output_file',type=str, help="Output csv location")
    parser.add_argument('-c', '--config', type=str , help='Configuration file')
    
    args = parser.parse_args()
    main(args)
    
