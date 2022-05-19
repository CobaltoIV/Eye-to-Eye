import pandas as pd
import datetime
import argparse
import cv2


class bounds:
    def __init__(self, x0, x1, y0, y1):
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


def doc_classification(res, x, y, doc, mode, date):
    """Function that alters classification given in function of doctor maneurisms during consultation
    """
    day = date['d']
    month = date['m']
    # Neurologia
    if doc == 'D1' and mode == 'Presential':
        if res == 'Left Of Screen' and y > 5.0:
            res = 'Keyboard'
        elif res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif doc == 'D8' and mode == 'Presential':
        if res == 'Left Of Screen' and y > 3.0:
            res = 'Keyboard'
        elif res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif doc == 'D2' and mode == 'Presential':
        if res == 'Left Of Screen' and x > -1.5 and y > 5.0:
            res = 'Keyboard'
        elif res == 'Left Of Screen' and x < -1.5 and y > 8.0:
            res = 'Keyboard'
        elif res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif doc == 'D11' and mode == 'Presential':
        if res == 'Left Of Screen' and y > 7.0:
            res = 'Keyboard'
        elif res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif doc == 'D5' and mode == 'Presential':
        if res == 'Right Of Screen' and y > 1.0:
            res = "Keyboard"
        elif res == 'Right Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Left Of Screen':
            res = 'Screen'
    elif doc == 'D7' and mode == 'Presential':
        if res == 'Right Of Screen' and x < 1 and y > 1.0:
            res == 'Keyboard'
        elif res == 'Right Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Left Of Screen':
            res = 'Screen'
    elif doc == 'D4' and mode == 'Presential':
        if res == 'Right Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Left Of Screen':
            res = 'Screen'
    elif doc == 'D16' and mode == 'Presential':
        if res == 'Right Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Left Of Screen':
            res = 'Screen'
    elif doc == 'D3' and mode == 'Presential':
        if y > 1.5:
            res = 'Keyboard'
        elif res == 'Right Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Left Of Screen':
            res = 'Screen'
    elif doc == 'D10' and mode == 'Presential':
        if y > 3:
            res = 'Keyboard'
        elif res == 'Right Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Left Of Screen':
            res = 'Screen'
    elif doc == 'D13' and mode == 'Presential':
        if res == 'Right Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Left Of Screen':
            res = 'Screen'
    elif doc == 'D15' and mode == 'Presential':
        if res == 'Right Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Left Of Screen':
            res = 'Screen'
    elif doc == 'D6' and mode == 'Presential':
        if y > 1.5:
            res = 'Keyboard'
        elif res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif doc == 'D12' and mode == 'Presential':
        if y > 1.8:
            res = 'Keyboard'
        elif res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif doc == 'D14' and mode == 'Presential':
        if y > 2.0:
            res = 'Keyboard'
        elif res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif doc == 'D9' and mode == 'Presential':
        if y > 2.5:
            res = 'Keyboard'
        elif res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif doc == 'D2' and mode == 'Virtual' and day == '04' and month == '03':
        if y > 3.0:
            res = 'Keyboard'
        elif res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif doc == 'D3' and mode == 'Virtual':
        if res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif doc == 'D8' and mode == 'Virtual' and day == '09' and month == '03':
        if y > 1.5 and x > -0.75:
            res = 'Keyboard'
        elif res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif doc == 'D9' and mode == 'Virtual':
        if res == 'Left Of Screen' and ((y > 1.5 and x > -0.5)):
            res = 'Keyboard'
        elif res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif doc == 'D10' and mode == 'Virtual':
        if res == 'Left Of Screen' and (y > 1.5 and x > -1.0):
            res = 'Keyboard'
        elif res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif doc == 'D12' and mode == 'Virtual':
        if res == 'Left Of Screen' and y > 2.0:
            res = 'Keyboard'
        elif res == 'Left Of Screen' and x > -0.5:
            res = 'Screen'
        elif res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif doc == 'D13' and mode == 'Virtual':
        if res == 'Left Of Screen' and y > 2.0:
            res = 'Keyboard'
        elif res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif doc == 'D14' and mode == 'Virtual' and (day == '29' or day == '22' or day == '25'):
        if y > 2.0:
            res = 'Keyboard'
        elif res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif doc == 'D14' and mode == 'Virtual':
        if res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif doc == 'D16' and mode == 'Virtual' and (day == '04' or day == '07' or day == '08' or day == '14' or day == '15' or day == '22' or day == '23'):
        if y > 2.0:
            res = 'Keyboard'
        elif res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif doc == 'D16' and mode == 'Virtual' and day == '21':
        if y > 1.3:
            res = 'Keyboard'
        elif res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif doc == 'D16' and mode == 'Virtual' and day == '29':
        if y > 3.0:
            res = 'Keyboard'
        elif res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif doc == 'D16' and mode == 'Virtual' and day == '30':
        if y > 2.0:
            res = 'Keyboard'
        elif res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'
    elif mode == "Virtual":
        if res == 'Left Of Screen' or res == 'not_in_frame':
            res = 'Patient'
        if res == 'Right Of Screen':
            res = 'Screen'

    return res


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


def add_columns(row, monitor_W, monitor_H, fps, gaze360_bounds, doc, mode, date):
    """Function applied to data to classify and process all samples
    """
    gaze360_x = row["gaze360_x"]
    gaze360_y = row["gaze360_y"]

    confidence = row["f_confidence"]
    frame = row["frame"]

    time = get_timestamp(frame, fps)

    gaze360_res = get_result(gaze360_x, gaze360_y, confidence, gaze360_bounds)

    gaze360_res = doc_classification(
        gaze360_res, gaze360_x, gaze360_y, doc, mode, date)

    gaze360_2d = [gaze360_x*monitor_W, gaze360_y*monitor_H]

    return time, gaze360_res, gaze360_2d[0], gaze360_2d[1]


def main(args):
    # Loading Screen Size
    if not args.config:
        raise Exception('Please specify configuration file.')
    fs = cv2.FileStorage(
        args.config, cv2.FILE_STORAGE_READ)
    W = fs.getNode("monitor_W").real()
    H = fs.getNode("monitor_H").real()
    gaze360_b = bounds(fs.getNode("x0_360").real(), fs.getNode(
        "x1_360").real(), fs.getNode("y0_360").real(), fs.getNode("y1_360").real())

    name = args.input_file

    if not args.output_file:
        args.output_file = f'{args.consult_folder}/proc_res/{name}_classified_gaze.csv'
        print('Output file not defined saving results to ' + args.output_file)
        totals_outfile = f'{args.consult_folder}/proc_res/Totals/{name}_gaze360_totals.csv'

    str = name.split('_')
    doc = str[0]
    timestamp = str[1].split('-')
    mode = timestamp[0]
    date = {'d': timestamp[1], 'm': timestamp[2], 'y': timestamp[3]}
    # Video fps
    fps = 15
    print(name)
    print(doc)
    print(mode)
    print(date)
    print(gaze360_b.x0)
    print(gaze360_b.x1)
    print(gaze360_b.y0)
    print(gaze360_b.y1)
    nm = ["frame", "f_found", "f_confidence", "facex", "facey", "facez",
          "2d_x", "2d_y", "3d_x", "3d_y", "3d_z"]

    # Load samples from .csv
    gaze_360_out = f'{args.consult_folder}/{name}_gaze360_out.csv'

    df_360 = pd.read_csv(gaze_360_out, index_col=0)

    df_combined = pd.DataFrame()

    df_combined['frame'] = df_360['frame']
    df_combined['f_confidence'] = df_360['f_confidence']
    df_combined['gaze360_x'] = df_360['2d_x']
    df_combined['gaze360_y'] = df_360['2d_y']

    df_combined['gaze360_3d_x'] = df_360['3d_x']
    df_combined['gaze360_3d_y'] = df_360['3d_y']
    df_combined['gaze360_3d_z'] = df_360['3d_z']

    df_combined[['timestamp', 'gaze360_res', '2d_gaze360_x', '2d_gaze360_y']] = df_combined.apply(
        add_columns, args=(W, H, fps, gaze360_b, doc, mode, date), axis=1, result_type='expand')

    df_combined[['timestamp', 'gaze360_res', 'gaze360_x',
                 'gaze360_y']].to_csv(args.output_file)

    doc_totals_csv = f'{args.consult_folder}/proc_res/Totals/Stats.csv'

    try:
        df_totals = pd.read_csv(
            f'{args.consult_folder}/proc_res/Totals/Stats.csv', index_col=0)
    except Exception:
        print('Stats.csv is empty, Create Dataframe')
        df_totals = pd.DataFrame()

    consult_stats = df_combined['gaze360_res'].value_counts()

    frame_total = consult_stats.sum()

    consult_stats_percentages = pd.Series(
        consult_stats/frame_total).add_suffix('_Percentage')

    consult_stats['Frame_total'] = consult_stats.sum()

    consult_stats['Name'] = name

    consult_stats = consult_stats.append(consult_stats_percentages)

    df_totals = df_totals.append(consult_stats, ignore_index=True)

    df_totals.to_csv(doc_totals_csv)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process gaze output csv file')

    parser.add_argument('-i', '--input_file', type=str, help="Input .mp4 name")
    parser.add_argument('-d', '--consult_folder', type=str,
                        help="Doctor and mode folder")
    parser.add_argument('-o', '--output_file', type=str,
                        help="Output csv location")
    parser.add_argument('-c', '--config', type=str, help='Configuration file')

    args = parser.parse_args()
    main(args)
