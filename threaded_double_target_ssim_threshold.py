import math
import os
import queue
import threading

import cv2
import numpy
from skimage.metrics import structural_similarity as ssim


class Worker(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(maxsize=20)

    def decode(self, video_path, target_frame, target_frame_next, fnos, worker_queue, stop_event, thread_number, frames_per_thread, result, no_match):
        self.queue.put((video_path, target_frame, target_frame_next, fnos, worker_queue,
                       stop_event, thread_number, frames_per_thread, result, no_match))

    def run(self):
        """the run loop to execute frame reading"""
        video_path, target_frame, target_frame_next, fnos, worker_queue, stop_event, thread_number, frames_per_thread, result, no_match = self.queue.get()
        cap = cv2.VideoCapture(video_path)
        # print(f'starting thread {thread_number}')
        # set initial frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnos[0])
        success, img = cap.read()
        # print(success)
        if success:
            idx = 1
            max_current_ssim_compare, max_current_ssim_compare_next = -1, -1
            match_found = False
            height, width, _ = img.shape
            dim = int(width*.1), int(height*.1)
            resize = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            resize = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
            current_frame1 = resize
            # print(thread_number, idx)
            while success and not stop_event.is_set():
                # print(thread_number, idx)
                success, img_next = cap.read()
                if success:
                    idx += 1
                    height_next, width_next, _ = img_next.shape
                    dim_next = int(width_next*.1), int(height_next*.1)
                    resize_next = cv2.resize(
                        img_next, dim_next, interpolation=cv2.INTER_AREA)
                    resize_next = cv2.cvtColor(resize_next, cv2.COLOR_BGR2GRAY)
                else:
                    break  # this break also handles trying to grab a frame after the video has ended

                current_ssim_compare = ssim(resize, target_frame)
                # print(current_ssim_compare,thread_number,(idx - 2) + frames_per_thread * thread_number)
                if current_ssim_compare > max_current_ssim_compare:
                    max_current_ssim_compare = current_ssim_compare
                    current_frame1 = resize
                    current_frame_number = (
                        idx - 2) + frames_per_thread * thread_number

                current_ssim_compare_next = ssim(
                    resize_next, target_frame_next)
                # print(current_ssim_compare_next,thread_number,(idx - 2) + frames_per_thread * thread_number+1)
                if current_ssim_compare_next > max_current_ssim_compare_next:
                    max_current_ssim_compare_next = current_ssim_compare_next
                    current_frame2 = resize_next

                if current_ssim_compare > .5 and current_ssim_compare_next > .5:
                    # calculate the current frame number
                    frame_number = (idx - 2) + \
                        frames_per_thread * thread_number
                    match_found = True
                    print(current_ssim_compare, current_ssim_compare_next)
                    worker_queue.put(thread_number)
                    result.extend([resize, resize_next, current_ssim_compare,
                                  current_ssim_compare_next, frame_number, match_found])
                    break

                if idx > len(fnos):  # use of > ensures ssim is calculated at boundary of fnos. without idx the while loop would continue reading past the length of fnos
                    break

                resize = resize_next
            # print(f'Thread {thread_number} exiting')
            # not stop_event.is_set() required to prevent unbound variable issue where one thread completes with a match before the others start
            if not match_found and not stop_event.is_set():
                no_match.append([current_frame1, current_frame2, max_current_ssim_compare,
                                max_current_ssim_compare_next, current_frame_number, match_found])
                # print(no_match)

            # put worker ID in queue
            # if not stop_event.is_set():
            #     worker_queue.put(thread_number)
        cap.release()


def threaded_double_target_ssim_threshold(video_path, target_frame, target_frame_next, frame_window=None):
    cap = cv2.VideoCapture(video_path)

    if frame_window is None:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fnos = list(range(total_frames))
    else:
        total_frames = int(frame_window[1]-frame_window[0])
        fnos = list(range(frame_window[0], frame_window[1]+1))

    n_threads = 4  # n_threads is the number of worker threads to read video frame
    # store frame number for each threads
    tasks = [[] for _ in range(n_threads)]
    frames_per_thread = math.ceil(len(fnos) / n_threads)

    for idx, fno in enumerate(fnos):
        tasks[math.floor(idx / frames_per_thread)].append(fno)

    # list to hold winner and no matches
    result = []
    no_match = []

    # queue for workers
    worker_queue = queue.Queue()

    # indicator for other threads to stop
    stop_event = threading.Event()

    # create and start threads
    threads = []
    for _ in range(0, n_threads):
        w = Worker()
        # w.setDaemon(True) # daemon threads will die once ssim threshold is reached since program ends
        threads.append(w)
        w.start()

    for idx, w in enumerate(threads):
        w.decode(video_path, target_frame, target_frame_next,
                 tasks[idx], worker_queue, stop_event, idx, frames_per_thread, result, no_match)

    # print(worker_queue.empty())
    while worker_queue.empty():
        if len(no_match) != 4:
            # print(len(no_match))
            pass
        else:
            worker_queue.put('No match!')
            current_max_index = 0
            current_max = no_match[0][2]
            for idx, item in enumerate(no_match):
                if item[2] > current_max:
                    current_max = item[2]
                    current_max_index = idx
            result.extend(no_match[idx])
            break

    # this will block until the first element is in the queue
    first_finished = worker_queue.get()
    # print(f'Thread {first_finished} was first!')
    # print(result[2])

    # signal the rest to stop working
    stop_event.set()

    # while True:
    #     cv2.imshow('first',result[0])
    #     cv2.imshow('second',result[1])
    #     if cv2.waitKey(1) == ord('q'):
    #         break

    # Adding code to search second video for match
    cap.release()
    return (tuple(result))
