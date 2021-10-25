import sys

import cv2
import numpy as np


def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'X-Coord: {x}, Y-Coord: {y}')
        param.append([float(x), float(y), 0.])


def get_vertices():
    real_vertices = []
    real_img = cv2.imread('resources/court_real.JPG')
    cv2.namedWindow('real', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('real', get_coordinates, real_vertices)

    top_vertices = []
    top_img = cv2.imread('resources/court-top.png')
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

    secondary_1 = np.array([[610., 560., 0.],
                          [610., -560., 0.],
                          [390., -560., 0.],
                          [390., 560., 0.]])

    # Pad the data with ones, so that our transformation can do translations too
    n = primary.shape[0]
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:,:-1]
    X = pad(primary)
    Y = pad(secondary)

    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A
    A, res, rank, s = np.linalg.lstsq(X, Y)

    transform = lambda x: unpad(np.dot(pad(x), A))

    primary_transformed = transform(primary)
    max_error = np.abs(secondary - primary_transformed).max()

    print(f'transformation-matrix:\n{A}')

    A[np.abs(A) < 1e-10] = 0  # set really small values to zero
    print(f'transformation-matrix with zeroes:\n{A}')

    print(f'Target:\n{secondary}')
    print(f'Result:\n{primary_transformed}')
    print(f'Max error:\n{max_error}')


if __name__ == "__main__":
    get_actual_values = sys.argv[1]
    if get_actual_values == '1':
        real_vertices, top_vertices = get_vertices()
    else:
        # order is top-left, bottom-left, top-right, bottom-right
        real_vertices = [[1077., 2662., 0.], [142., 2870., 0.], [4409., 2676., 0.], [5378., 2870., 0.]]
        top_vertices = [[504., 363., 0.], [500., 4208., 0.], [7617., 363., 0.], [7617., 4204., 0.]]
    get_transformation_matrix(real_vertices, top_vertices)

