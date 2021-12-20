import math
from typing import Union

import cv2 as cv
import numpy as np
# import ball_net as bn
import sys

count = 0

R = 60
EPS = 1e-6
EPS2 = 0.5

STATUS_INIT = 0
STATUS_STATIC = 1
STATUS_DIRECTED = 2


def pt_dist(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)


class Blob:
    count = 1

    def __init__(self, x, y, radius, age):
        self.id = Blob.count
        Blob.count += 1
        self.points = [[x, y]]
        self.point_properties = [[radius, age]]
        self.status = STATUS_INIT
        self.age = age
        self.nx = None
        self.ny = None

    def fit(self, x, y):
        # get the distance from the last added point (x and y) to another Point x and y
        d = pt_dist(self.points[-1][0], self.points[-1][1], x, y)
        return d < R, d

    def add(self, x, y, r, a):
        self.points.append([x, y])
        self.point_properties.append([r, a])
        self.age = a
        if len(self.points) > 2:
            # if self.status == STATUS_DIRECTED and self.nx is not None:
            #  print("Predict", self.nx, self.ny, "vs", x, y)

            dx1 = self.points[-2][0] - self.points[-3][0]
            dy1 = self.points[-2][1] - self.points[-3][1]

            dx2 = x - self.points[-2][0]
            dy2 = y - self.points[-2][1]

            d1 = pt_dist(self.points[-2][0], self.points[-2][1], x, y)
            d2 = pt_dist(self.points[-2][0], self.points[-2][1], self.points[-3][0], self.points[-3][1])
            if dx1 * dx2 > 0 and dy1 * dy2 > 0 and d1 > 5 and d2 > 5:
                self.status = STATUS_DIRECTED
                # print("Directed", self.pts)
                # self.predict()
            elif self.status != STATUS_DIRECTED:
                self.status = STATUS_STATIC

    def predict(self):
        npts = np.array(self.points)
        l = len(self.points) + 1
        idx = np.array(range(1, l))

        kx = np.polyfit(idx, npts[:, 0], 1)
        fkx = np.poly1d(kx)

        ky = np.polyfit(idx, npts[:, 1], 1)
        fky = np.poly1d(ky)

        self.nx = fkx(l)
        self.ny = fky(l)
        return self.nx, self.ny


Blobs = []
ball_blob: Union[Blob, None] = None
prev_ball_blob: Union[Blob, None] = None


def get_ball_blob():
    return ball_blob


def find_closest_existing_blob(center_x, center_y):
    global Blobs, count
    rbp = []
    sbp = []

    for blob in Blobs:
        # its fitting if the distance is below 60 (why 60?)
        fit, distance = blob.fit(center_x, center_y)
        if fit:
            # new blob is not longer than 4 blobs away
            if count - blob.age < 4:
                rbp.append([blob, distance])
            elif blob.status == STATUS_STATIC:
                sbp.append([blob, distance])

    if len(sbp) + len(rbp) == 0:
        return None
    if len(rbp) > 0:
        # sort by distance
        rbp.sort(key=lambda e: e[1])
        # return blob with the lowest distance
        return rbp[0][0]
    else:
        # sort by distance
        sbp.sort(key=lambda e: e[1])
        return sbp[0][0]


def handle_blob(center_x, center_y, radius):
    global Blobs, count, ball_blob
    blob = find_closest_existing_blob(center_x, center_y)
    if blob is None:
        Blobs.append(Blob(center_x, center_y, radius, count))
        return
    blob.add(center_x, center_y, radius, count)
    if blob.status == STATUS_DIRECTED:
        if not ball_blob:
            ball_blob = blob
        # if the current blob has more data its the new ball blob
        elif len(blob.points) > len(ball_blob.points):
            ball_blob = blob


def begin_gen():
    global ball_blob, prev_ball_blob
    prev_ball_blob = ball_blob
    ball_blob = None


def end_gen():
    global count, ball_blob
    count += 1


def handle_blobs(mask, frame):
    contours, _ = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    # detect_blobs_in_mask(frame)
    begin_gen()
    for contour in contours:
        rectangle_origin_x, rectangle_origin_y, rectangle_width, rectangle_height = cv.boundingRect(contour)

        # cut the rectangle that is bounding the blob out of the mask
        cut_blob_from_mask = mask[
                             rectangle_origin_y: rectangle_origin_y + rectangle_height,
                             rectangle_origin_x: rectangle_origin_x + rectangle_width]

        # cut the bounding rectangle from the frame
        cut_frame = frame[
                    rectangle_origin_y: rectangle_origin_y + rectangle_height,
                    rectangle_origin_x: rectangle_origin_x + rectangle_width]

        # cv.imshow("Cut-Blob", cut_blob_from_mask)
        # cv.imshow("Cut-Frame", cut_frame)

        if not is_valid_ball(cut_blob_from_mask, rectangle_height, rectangle_width):
            # cv.imshow("Cut-Blob", cut_blob_from_mask)
            # cv.imshow("Cut-Frame", cut_frame)
            # print("blob filtered out")
            # cv.waitKey(0)
            destroy_blobber_windows()
            continue
        # cv.imshow("Cut-Frame", cut_frame)

        # why is this done here, whats the benefit?
        # so only the real detected blob is there not the noise from cutting?
        cut_c = cv.bitwise_and(cut_frame, cut_frame, mask=cut_blob_from_mask)
        # cv.imshow("Cut-C", cut_c)
        # cv.imshow("Cut-Blob", cut_blob_from_mask)
        # cv.imshow("Cut-Frame", cut_frame)
        # print("blob was allowed")
        # cv.waitKey(0)



        # get data (coordinates) for the enclosing circle of the detected ball
        destroy_blobber_windows()
        ((x, y), radius) = cv.minEnclosingCircle(contour)

        # find out if the blob is directed with a previous blob and also add it to blob list
        handle_blob(int(x), int(y), int(radius))

    end_gen()


def get_polygon_curve_vertices(contour):
    arc_length = cv.arcLength(contour, True)
    print(f'arc {arc_length}')
    # approx = cv.approxPolyDP(contour, 0.5, True)
    approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
    polys = len(approx)
    return polys


def destroy_blobber_windows():
    if cv.getWindowProperty("Cut-Blob", cv.WND_PROP_VISIBLE) == 1.0:
        cv.destroyWindow("Cut-Blob")

    if cv.getWindowProperty("Cut-Frame", cv.WND_PROP_VISIBLE) == 1.0:
        cv.destroyWindow("Cut-Frame")

    if cv.getWindowProperty("Cut-C", cv.WND_PROP_VISIBLE) == 1.0:
        cv.destroyWindow("Cut-C")


def is_valid_ball(blob, bounding_rect_height, bounding_rect_width):
    rectangle_shorter_side = min(bounding_rect_width, bounding_rect_height)
    rectangle_longer_side = max(bounding_rect_width, bounding_rect_height)
    rectangle_ratio = rectangle_longer_side / rectangle_shorter_side
    print(f'ratio: {rectangle_ratio}, short: {rectangle_shorter_side}, long: {rectangle_longer_side}')

    # actual ball sides are around 7-10
    if rectangle_shorter_side < 5 or rectangle_longer_side > 13 or rectangle_ratio > 1.5:
        print("blob sizes are wrong")
        # print(f'ratio: {rectangle_ratio}, short: {rectangle_shorter_side}, long: {rectangle_longer_side}')
        return False

    is_blob, amount_of_non_zeroes = check_blob(blob, bounding_rect_width, bounding_rect_height)
    # cv.imshow('Mask', mask)
    # if not is_blob:
    #     print("blob to many zeroes")
    #     return False

    probability_non_zeroes = amount_of_non_zeroes / (bounding_rect_width * bounding_rect_height)
    # at least half of the pixels should be non-zeroes
    if probability_non_zeroes < 0.5:
        print("blob to few non-zeroes")
        return False

    # additional checks
    # is the blob a ball? Decided by the NN
    # prediction = bn.check_pic(cut_c)
    # if prediction <= 0.5:
    #     # cv.destroyWindow("Cut-Blob")
    #     # cv.destroyWindow("Cut-Frame")
    #     # cv.destroyWindow("Cut-C")
    #     print(f'no-ball: {prediction}')
    #     continue
    # print(f'ball: {prediction}')

    # count polygons
    # polys = get_polygon_curve_vertices(contour)
    # print(f'number of polys: {polys}')
    # # if polys < 8 or polys > 13:
    # if polys < 10:
    #     print('no circle')
    #     continue
    # print('circle')
    #
    # calculate properties of circle
    # if not is_contour_a_circle(contour):
    #     continue

    return True


def is_contour_a_circle(contour):
    area = cv.contourArea(contour)
    print(f'contour area: {area}')
    perimeter = cv.arcLength(contour, True)
    circularity = 4*math.pi*(area/(perimeter*perimeter))
    print(f'circularity: {circularity}')
    return circularity >= 0.85

def detect_blobs_in_mask(mask):
    detector = cv.SimpleBlobDetector_create()
    # Detect blobs.
    keypoints = detector.detect(mask)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv.drawKeypoints(mask, keypoints, np.array([]), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Show keypoints
    cv.imshow("Keypoints", im_with_keypoints)
    # cv.waitKey(0)



def check_blob(blob, width, height):
    # x and y are always 0
    origin_x = 0
    origin_y = 0

    dx = int(width / 5)
    x0 = origin_x + 2 * dx
    vertical_part_of_blob = blob[origin_y: origin_y + height, x0: x0 + dx]

    dy = int(height / 5)
    y0 = origin_y + 2 * dy
    # this cuts the 3rd fifth part (horizontal) out from the blob
    horizontal_part_of_blob = blob[y0: y0 + dy, origin_x: origin_x + width]

    non_zeroes_in_horizontal_strip = cv.countNonZero(horizontal_part_of_blob)
    non_zeroes_in_vertical_strip = cv.countNonZero(vertical_part_of_blob)
    non_zeroes_in_blob = cv.countNonZero(blob)

    lower_count_of_non_zeroes = min(non_zeroes_in_horizontal_strip, non_zeroes_in_vertical_strip)
    upper_count_of_non_zeroes = max(non_zeroes_in_horizontal_strip, non_zeroes_in_vertical_strip)

    if lower_count_of_non_zeroes > 0:
        ratio_of_non_zeroes_in_both_strips = upper_count_of_non_zeroes / lower_count_of_non_zeroes
    else:
        ratio_of_non_zeroes_in_both_strips = 1000

    ratio_of_non_zeroes_for_horizontal_and_blob = non_zeroes_in_horizontal_strip / non_zeroes_in_blob
    ratio_of_non_zeroes_for_vertical_and_blob = non_zeroes_in_vertical_strip / non_zeroes_in_blob

    # what are these ratios? why 1.5, 0.15
    return \
        ratio_of_non_zeroes_in_both_strips < 1.5 and \
        ratio_of_non_zeroes_for_horizontal_and_blob > 0.15 and \
        ratio_of_non_zeroes_for_vertical_and_blob > 0.15, non_zeroes_in_blob


def draw_ball(pic):
    ball = get_ball_blob()
    if ball is not None:
        cv.circle(pic, (ball.points[-1][0], ball.points[-1][1]), 10, (0, 200, 0), 3)
    else:
        if prev_ball_blob is not None:
            x, y = prev_ball_blob.predict()
            cv.circle(pic, (int(x), int(y)), 10, (0, 200, 0), 3)


found_points = []
def draw_ball_path(pic):
    ball = get_ball_blob()
    # try detection with vectors and their direction (so 4 points)
    if ball is not None:
        # points_iterator = iter(ball.points)
        sub_points_size = 4
        points = ball.points
        for index, point_to_draw in enumerate(points):
            # point_to_draw = ball.points[index]
            next_four_points = ball.points[index:index+sub_points_size]
            # print(f'current point: {point_to_draw}')
            # print(f'next four points {next_four_points}')
            # next_two_points = list(itertools.islice(points_iterator, 2))
            if len(next_four_points) == 4:
                intersection = get_intersect(next_four_points[0], next_four_points[1], next_four_points[2], next_four_points[3])
                y_coordinates = map(lambda point: point[1], next_four_points)
                intersection_y = intersection[1]

                if (intersection_y < float('inf')) and all(i <= intersection_y for i in y_coordinates):
                    # print(f'lowest point found: {intersection}')
                    cv.circle(pic, (intersection[0], intersection_y), 3, (0, 0, 255), -1)
            cv.circle(pic, (point_to_draw[0], point_to_draw[1]), 3, (150, 150, 150), -1)


def get_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (int(x/z), int(y/z))


def draw_blobs(w, h):
    pic = np.zeros((h, w, 3), np.uint8)
    for b in Blobs:
        clr = (200, 200, 200)
        if b.status == STATUS_STATIC:
            clr = (0, 200, 0)
        elif b.status == STATUS_DIRECTED:
            clr = (200, 0, 0)
            # if b.v is not None:
            #     cv.line(pic, (b.points[0][0], b.points[0][1]), (b.points[-1][0], b.points[-1][1]), (255, 0, 0), 1)
        for p in b.points:
            cv.circle(pic, (p[0], p[1]), 3, clr, -1)

    draw_ball(pic)

    return pic


def test_clip(path):
    vs = cv.VideoCapture(path)
    backSub = cv.createBackgroundSubtractorMOG2()
    n = 0
    while (True):
        ret, frame = vs.read()
        if not ret or frame is None:
            break

        h = int(frame.shape[0] / 2)
        w = int(frame.shape[1] / 2)

        frame = cv.resize(frame, (w, h))
        mask = backSub.apply(frame)

        mask = cv.dilate(mask, None)
        mask = cv.GaussianBlur(mask, (15, 15), 0)
        ret, mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

        handle_blobs(mask, frame)
        pic = draw_blobs(w, h)
        cv.imshow('frame', pic)
        cv.imwrite("frames/frame-{:03d}.jpg".format(n), pic)
        if cv.waitKey(10) == 27:
            break
        n += 1


if __name__ == "__main__":
    test_clip(sys.argv[1])
    # test_clip("D:/Videos/aus4.avi")
