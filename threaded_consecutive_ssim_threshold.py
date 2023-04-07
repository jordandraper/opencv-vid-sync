import math
import os
import queue
import threading
from datetime import timedelta

import cv2
import numpy
from skimage.metrics import structural_similarity as ssim


def frame_ssim_compare(input1, input2):
    if os.path.isfile(input1) and os.path.isfile(input2):
        original = cv2.imread(input1)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        alternate = cv2.imread(input2)
        alternate = cv2.cvtColor(alternate, cv2.COLOR_BGR2GRAY)

    elif isinstance(input1, numpy.ndarray) and isinstance(input2, numpy.ndarray):
        original = cv2.cvtColor(input1, cv2.COLOR_BGR2GRAY)
        alternate = cv2.cvtColor(input2, cv2.COLOR_BGR2GRAY)

    s = ssim(original, alternate)
    return s

def transform_frame(frame,scaling_factor=.1):
    height, width, _ = frame.shape
    dim = int(width*scaling_factor), int(height*scaling_factor)
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    grey_scale_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    return grey_scale_frame

def find_transition(video_path):
    cap = cv2.VideoCapture(video_path)
    idx = 0 
    ssim_scores = {}

    while True:
        _, frame_1 = cap.read()
        success, frame_2 = cap.read()
        if success:
            resized_frame_1 = transform_frame(frame_1)
            resized_frame_2 = transform_frame(frame_2)
            ssim_score = ssim(resized_frame_1, resized_frame_2)
            ssim_scores[idx] = ssim_score
            idx += 1
        else:
            break
    
    print(ssim_scores)
    min_index = min(ssim_scores, key=ssim_scores.get)
    print(min_index, ssim_scores[min_index])

class Worker(threading.Thread):
    """
    Specialized subclass to handle video through threads
    """
    def __init__(self):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(maxsize=20)

    def decode(self, video_path, fnos, worker_queue, stop_event, thread_number, frames_per_thread, result):
        self.queue.put((video_path, fnos, worker_queue, stop_event,
                       thread_number, frames_per_thread, result))
    
    def run(self):
        """the run loop to execute frame reading"""
        video_path, fnos, worker_queue, stop_event, thread_number, frames_per_thread, result = self.queue.get()
        cap = cv2.VideoCapture(video_path)

        # set initial frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnos[0])
        success, img = cap.read()

        ssim_scores = {}
        for fno in fnos:
            print(fno)
            if not stop_event.is_set():
                cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
                _, frame_1 = cap.read()
                success, frame_2 = cap.read()
                if success:
                    resized_frame_1 = transform_frame(frame_1)
                    resized_frame_2 = transform_frame(frame_2)
                    ssim_score = ssim(resized_frame_1, resized_frame_2)
                    ssim_scores[(fno,fno+1)] = ssim_score
                    # print(min(ssim_scores, key=ssim_scores.get))

                if ssim_score > 1  and cv2.countNonZero(resized_frame_1) and cv2.countNonZero(resized_frame_2):
                    worker_queue.put(fno)
                    print(f"This is thread {self}")
                    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    result.extend(
                        [resized_frame_1, resized_frame_2, ssim_score, fno,cap.get(cv2.CAP_PROP_POS_MSEC)])
                    break
        worker_queue.put(thread_number)
                # if idx > len(fnos):  # use of > ensures ssim is calculated at boundary of fnos. without idx the while loop would continue reading past the length of fnos
                #     break

                # resize = resize_next
            # put worker ID in queue
        # if not stop_event.is_set():
        #     worker_queue.put(thread_number)
        cap.release()


def threaded_consecutive_ssim_threshold(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fnos = list(range(total_frames))

    n_threads = 4  # n_threads is the number of worker threads to read video frame
    # store frame number for each threads
    tasks = [[] for _ in range(n_threads)]
    frames_per_thread = math.ceil(total_frames / n_threads)
    for idx, fno in enumerate(fnos):
        tasks[math.floor(idx / frames_per_thread)].append(fno)

    # list to hold winner
    result = []

    # queue for workers
    worker_queue = queue.Queue()

    # indicator for other threads to stop
    stop_event = threading.Event()

    # create and start threads
    threads = []
    for _ in range(n_threads):
        w = Worker()
        # w.setDaemon(True) # daemon threads will die once ssim threshold is reached since program ends
        threads.append(w)
        w.start()

    for idx, w in enumerate(threads):
        w.decode(video_path, tasks[idx], worker_queue,
                 stop_event, idx, frames_per_thread, result)

    # this will block until the first element is in the queue
    first_finished = worker_queue.get()
    print(f'Thread {first_finished} was first!')
    td = timedelta(seconds=result[4]/1000)
    print(f'SSIM score of {result[2]}, Frame number {result[3]}, Time: {td}')

    # signal the rest to stop working
    stop_event.set()

    while True:
        cv2.imshow('first',result[0])
        cv2.imshow('second',result[1])
        if cv2.waitKey(1) == ord('q'):
            break

    # Adding code to search second video for match
    cap.release()
    return (tuple(result))

# threaded_consecutive_ssim_threshold("./video.mp4")
find_transition("./video.mp4")