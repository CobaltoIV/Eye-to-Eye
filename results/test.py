import pandas as pd
import datetime
import argparse

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
    if confidence <= 0.80:  # OpenFace couldn't detect a face
        return "not_in_frame"
    elif x < boundaries.x1 and x > boundaries.x0 and y > 1:
        return "Keyboard"
    elif x > boundaries.x1 or y > boundaries.y1 or x < boundaries.x0 or y < boundaries.y0:  # Gaze outside screen boundaries
        return "Other"
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


def add_columns(row, monitor_W, monitor_H, fps, screen_bounds):
    """Function applied to data to classify and process all samples
    """
    x = row["2d_x"]
    y = row["2d_y"]
    
    confidence = row["f_confidence"]
    frame = row["frame"]
    
    time = get_timestamp(frame, fps)

    res = get_result(x, y, confidence, screen_bounds)

    coords = get_coord_screen(res, x, y, monitor_W, monitor_H)

    return [res, time, coords[0], coords[1]]


def main(args):
        # Loading Screen Size
    fs = cv2.FileStorage(
       args.config, cv2.FILE_STORAGE_READ)
    W = fs.getNode("monitor_W").real()
    H = fs.getNode("monitor_H").real()
    
    if not args.output_file:
        args.output_file = 'proc_res/'+args.input_file.replace('.csv', '_res.csv')
        print('Output file ot defined saving results to ' + args.output_file)
        
    screen_bounds = bounds(-0.25, 1.25, -0.25, 1)
    # Video fps
    fps = 15

    nm = ["frame", "f_found", "f_confidence", "facex", "facey", "facez",
        "2d_x", "2d_y", "3d_x", "3d_y", "3d_z"]

    # Load samples from .csv
    if "gaze360" in args.input_file:
        df = pd.read_csv(args.input_file,index_col=0)
    else:    
        df = pd.read_csv(args.input_file, names=nm)
    # Process and classify frames
    df[["res", "timestamp", "screen_x", "screen_y", "2d_x", "2d_y"]] = df.apply(
        add_columns, args=(W, H, fps, screen_bounds), axis=1, result_type='expand')

    # Save them in results.csv
    df[["frame", "timestamp", "res", "screen_x","screen_y"]].to_csv(args.output_file)
    
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process gaze output csv file')
    
    parser.add_argument('-i', '--input_file',type=str, help="Input csv location")
    parser.add_argument('-o', '--output_file',type=str, help="Output csv location")
    parser.add_argument('-c', '--config', type=str , help='Configuration file', default="/home/vislab/RecordingSetup/Calib/openfiles/my_pc.yml")
    
    args = parser.parse_args()
    main(args)
    
