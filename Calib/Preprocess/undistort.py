import numpy as np
import cv2 as cv
import json as json
import glob

with open('../InternalParameters/calib_1280.json') as f:
    j = json.load(f)
    mtx = np.array(j["camera_matrix"])
    dist = np.array(j['distortion_coefficients'])

images = glob.glob('ext*.jpg')

for fname in images:
    img = cv.imread(fname)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))
    # undistot
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(fname.replace('ext', 'input'), dst)

cv.destroyAllWindows()
