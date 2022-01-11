import queue
import sys
import threading

import cv2

import blobber
import cvlog as log
import calibration

frame_queue = queue.Queue()


def read_frames_into_queue(path):
    print("start receive")
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("reading failed")
            break

        frame_queue.put(frame)
    print("reading stopped")


def process_frames_from_queue():

    # open the logging file
    coordinates_file = sys.argv[2]
    calibration.initialise(coordinates_file)
    top_view = cv2.imread('resources/calibration/court-top-960.png')
    # run as long as frames are in queue
    print("start processing")
    background_subtraction = cv2.createBackgroundSubtractorMOG2()
    # frame_writer = cv2.VideoWriter('traced.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (640, 340))
    # mask_writer = cv2.VideoWriter('mask.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (640, 340))
    cv2.imshow('top_view', top_view)

    while not frame_queue.empty():
        # ret, frame = vs.read()
        # get frame from queue
        frame = frame_queue.get()

        if frame is None:
            print("frame not found")
            break
        # frame is numpy array
        mask, frame = preprocess_frame(frame, background_subtraction)
        # cv2.imwrite("./masks/mask-{:03d}.jpg".format(n), mask)
        # cv2.imwrite("./frames/frame-{:03d}.jpg".format(n), frame)

        # fit circle
        blobber.handle_blobs(mask, frame)
        blobber.draw_ball_path(frame, top_view)
        blobber.draw_ball(frame)
        log.image(log.Level.TRACE, frame)
        cv2.imshow('side_view', frame)
        cv2.imshow('top_view', top_view)
        # cv2.waitKey(0)

        # frame_writer.write(frame)
        # mask_writer.write(mask)
        if cv2.waitKey(10) == 27:
            break
    print("stopping processing")
    destroy_main_windows()
    # cv2.destroyWindow("dilate")
    # cv2.destroyWindow("backsub")
    # cv2.destroyWindow("opening")
    # frame_writer.release()
    # mask_writer.release()


def preprocess_frame(frame, back_sub):
    h = frame.shape[0]
    w = frame.shape[1]
    frame = cv2.resize(frame, (int(w / 2), int(h / 2)))
    back_subbed = back_sub.apply(frame)
    # mask = background_subtraction.apply(frame, learningRate=0.001)
    # cv2.imshow('backsub', back_subbed)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(back_subbed, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('opening', opened)

    # dilates using kernel with a default 3x3 matrix
    #     https://docs.opencv.org/4.5.1/db/df6/tutorial_erosion_dilatation.html
    dilated = cv2.dilate(opened, None, iterations=1)
    # cv2.imshow('dilate', dilated)
    blurred = cv2.GaussianBlur(dilated, (15, 15), 0)
    # cv2.imshow('blur', blurred)
    # log.image(log.Level.TRACE, blurred)

    ret, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    # ret, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow('threshold', thresholded)
    log.image(log.Level.TRACE, thresholded)

    return thresholded, frame


def destroy_main_windows():
    if cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) == 1.0:
        cv2.destroyWindow("result")

    if cv2.getWindowProperty("top_view", cv2.WND_PROP_VISIBLE) == 1.0:
        cv2.destroyWindow("top_view")

    if cv2.getWindowProperty("threshold", cv2.WND_PROP_VISIBLE) == 1.0:
        cv2.destroyWindow("threshold")

    if cv2.getWindowProperty("blur", cv2.WND_PROP_VISIBLE) == 1.0:
        cv2.destroyWindow("blur")

    if cv2.getWindowProperty("mask", cv2.WND_PROP_VISIBLE) == 1.0:
        cv2.destroyWindow("mask")

    if cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) == 1.0:
        cv2.destroyWindow("frame")

    if cv2.getWindowProperty("bounce", cv2.WND_PROP_VISIBLE) == 1.0:
        cv2.destroyWindow("bounce")


if __name__ == "__main__":
    source = sys.argv[1]

    reading_thread = threading.Thread(target=read_frames_into_queue, args=(source,))
    # start processing thread after 2 seconds, so queue has time to fill up
    processing_thread = threading.Timer(5, process_frames_from_queue)
    reading_thread.start()
    processing_thread.start()
    # read_frames_into_queue(video_source)
    # process_frames_from_queue()
