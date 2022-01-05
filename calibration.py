import sys

import cv2
import numpy as np


def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'X-Coord: {x}, Y-Coord: {y}')
        param.append([x, y])
        # param.append([float(x), float(y), 0.])


def get_vertices(real_path, top_path):
    real_vertices = []
    real_img = cv2.imread(real_path)
    # real_img = cv2.imread('resources/calibration/hd-marked.jpg')
    cv2.namedWindow('real', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('real', get_coordinates, real_vertices)

    top_vertices = []
    top_img = cv2.imread(top_path)
    # top_img = cv2.imread('resources/calibration/court-top.png')
    cv2.namedWindow('top', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('top', get_coordinates, top_vertices)

    while(True):
        cv2.imshow('real', real_img)
        cv2.imshow('top', top_img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    print(f'real: {real_vertices}')
    print(f'top: {top_vertices}')
    return real_vertices, top_vertices


def get_transformation_matrix(real_vertices, top_vertices):
    primary = np.array(real_vertices)
    secondary = np.array(top_vertices)

    # Pad the data with ones, so that our transformation can do translations too
    n = primary.shape[0]
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:,:-1]
    X = pad(primary)
    Y = pad(secondary)

    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A
    A, res, rank, s = np.linalg.lstsq(X, Y)

    A[np.abs(A) < 1e-10] = 0  # set really small values to zero

    transform = lambda x: unpad(np.dot(pad(x), A))

    primary_transformed = transform(primary)
    max_error = np.abs(secondary - primary_transformed).max()

    print(f'transformation-matrix:\n{A}')

    print(f'transformation-matrix with zeroes:\n{A}')

    print(f'Target:\n{secondary}')
    print(f'Result:\n{primary_transformed}')
    print(f'Max error:\n{max_error}')


def get_opencv_transformation_matrix(real_vertices, top_vertices):
    primary = np.float32(real_vertices)
    secondary = np.float32(top_vertices)

    matrix = cv2.getAffineTransform(primary, secondary)

    fst_prim_pt = np.float32([primary[0]])
    custom_point = np.float32([[127, 388]])
    primary_transformed = cv2.transform(np.array([custom_point]), matrix)[0]

    print(f'transformation-matrix:\n{matrix}')

    print(f'Target:\n{secondary}')
    print(f'Result:\n{primary_transformed}')


if __name__ == "__main__":
    get_actual_values = sys.argv[1]
    if get_actual_values == 'pic':
        real_pic_path = sys.argv[2]
        top_pic_path = sys.argv[3]
        real_vertices, top_vertices = get_vertices(real_pic_path, top_pic_path)
    else:

        # order is top-left, bottom-left, top-right
        real_vertices = [[199, 376], [30, 401], [763, 380]]
        # real_vertices = [[1133.0, 2137.0, 0.0], [178.0, 2282.0, 0.0], [4351.0, 2163.0, 0.0], [5300.0, 2319.0, 0.0]]
        top_vertices = [[508, 358], [508, 4208], [7621, 363]]
        # top_vertices = [[508.0, 358.0, 0.0], [508.0, 4208.0, 0.0], [7621.0, 363.0, 0.0], [7617.0, 4204.0, 0.0]]
    # get_transformation_matrix(real_vertices, top_vertices)
    get_opencv_transformation_matrix(real_vertices, top_vertices)

