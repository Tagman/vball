import json
import sys

import cv2
import numpy as np


matrix = None

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


def get_transformation_matrix(source_points, target_points):
    global matrix
    primary = np.float32(source_points)
    secondary = np.float32(target_points)

    matrix = cv2.getAffineTransform(primary, secondary)

    print(f'transformation-matrix:\n{matrix}')


def initialise(coordinates_file_path):
    with open(coordinates_file_path) as coordinates_file:
        coordinates = json.load(coordinates_file)

    source_coordinates = coordinates['source']
    target_coordinates = coordinates['target']

    get_transformation_matrix(source_coordinates, target_coordinates)


def transform(point):
    source_point = np.float32([point])
    transformed_point = cv2.transform(np.array([source_point]), matrix)[0]
    return transformed_point[0]


if __name__ == "__main__":
    get_actual_values = sys.argv[1]
    if get_actual_values == 'pic':
        real_pic_path = sys.argv[2]
        top_pic_path = sys.argv[3]
        real_vertices, top_vertices = get_vertices(real_pic_path, top_pic_path)
    else:
        # order is top-left, bottom-left, top-right
        real_vertices = [[199, 376], [30, 401], [930, 407]]
        # real_vertices = [[1133.0, 2137.0, 0.0], [178.0, 2282.0, 0.0], [4351.0, 2163.0, 0.0], [5300.0, 2319.0, 0.0]]
        top_vertices = [[60, 43], [61, 504], [914, 504]]
        # top_vertices = [[508.0, 358.0, 0.0], [508.0, 4208.0, 0.0], [7621.0, 363.0, 0.0], [7617.0, 4204.0, 0.0]]
    # get_transformation_matrix(real_vertices, top_vertices)
    get_transformation_matrix(real_vertices, top_vertices)

