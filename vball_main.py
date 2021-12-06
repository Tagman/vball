import queue
import sys
import threading

import cv2

import blobber

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
    # run as long as frames are in queue
    print("start processing")
    background_subtraction = cv2.createBackgroundSubtractorMOG2()
    # frame_writer = cv2.VideoWriter('traced.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (640, 340))
    # mask_writer = cv2.VideoWriter('mask.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (640, 340))
    n = 0
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
        n = n + 1
        # fit circle
        blobber.handle_blobs(mask, frame)
        blobber.draw_ball_path(frame)
        blobber.draw_ball(frame)
        cv2.imshow('frame', frame)
        # cv2.waitKey(0)

        # frame_writer.write(frame)
        # mask_writer.write(mask)
        if cv2.waitKey(10) == 27:
            break
    print("stopping processing")
    cv2.destroyWindow("threshold")
    cv2.destroyWindow("frame")
    # cv2.destroyWindow("Keypoints")
    # cv2.destroyWindow("dilate")
    cv2.destroyWindow("blur")
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
    dilated = cv2.dilate(opened, None)
    # cv2.imshow('dilate', dilated)
    blurred = cv2.GaussianBlur(dilated, (15, 15), 0)
    cv2.imshow('blur', blurred)

    ret, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    # ret, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('threshold', thresholded)
    return thresholded, frame


def debug_blobs(frame, mask):
    blobber.handle_blobs(mask, frame)
    blobber.draw_ball_path(frame)
    blobber.draw_ball(frame)


if __name__ == "__main__":
    type_of_source = sys.argv[1]

    if type_of_source == "debug":
        name_of_debug_source = sys.argv[2]
        frame_name = name_of_debug_source + "_frame.jpg"
        mask_name = name_of_debug_source + "_mask.jpg"

        input_frame = cv2.imread(frame_name)
        input_mask = cv2.imread(mask_name)
        debug_blobs(input_frame, input_mask)

    elif type_of_source == "vid":
        source = sys.argv[2]
        reading_thread = threading.Thread(target=read_frames_into_queue, args=(source,))
        # start processing thread after 2 seconds, so queue has time to fill up
        processing_thread = threading.Timer(5, process_frames_from_queue)
        reading_thread.start()
        processing_thread.start()
        # read_frames_into_queue(video_source)
        # process_frames_from_queue()
