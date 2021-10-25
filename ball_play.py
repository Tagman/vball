import cv2 as cv
import blobber
import sys

def test_clip(path):
    vs = cv.VideoCapture(path)
    backSub = cv.createBackgroundSubtractorMOG2()
    n = 0
    video_writer = cv.VideoWriter('traced.avi', cv.VideoWriter_fourcc(*'DIVX'), 30, (960, 540))
    while (True):
        ret, frame = vs.read()
        if not ret or frame is None:
            break
        # frame is numpy array
        h = frame.shape[0]
        w = frame.shape[1]

        frame = cv.resize(frame, (int(w / 2), int(h / 2)))
        mask = backSub.apply(frame)

        # dilates using kernel with a default 3x3 matrix
        # https://docs.opencv.org/4.5.1/db/df6/tutorial_erosion_dilatation.html
        mask = cv.dilate(mask, None)
        mask = cv.GaussianBlur(mask, (15, 15), 0)
        ret, mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        # print(f'mask after thresholding: {mask}')
        # cv.imshow("threshold", mask)
        blobber.handle_blobs(mask, frame)

        blobber.draw_ball_path(frame)
        blobber.draw_ball(frame)
        # cv.imwrite("./frames/frame-{:03d}.jpg".format(n), frame)
        cv.imshow('frame', frame)
        video_writer.write(frame)
        if cv.waitKey(10) == 27:
            break
        n += 1
    video_writer.release()
    exit(0)


# test_clip("D:/Videos/aus4.avi")
test_clip(sys.argv[1])
