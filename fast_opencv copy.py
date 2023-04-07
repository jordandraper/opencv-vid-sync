import math
import os
import queue
import threading

import cv2
import numpy
from skimage.metrics import structural_similarity as ssim

from timer import Timer


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


video_path = ''
cap = cv2.VideoCapture(video_path)
sample_rate = 1
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fnos = list(range(0, total_frames, sample_rate))
n_threads = 4  # n_threads is the number of worker threads to read video frame
# store frame number for each threads
tasks = [[] for _ in range(0, n_threads)]
frame_per_thread = math.ceil(len(fnos) / n_threads)

tid = 0
for idx, fno in enumerate(fnos):
    tasks[math.floor(idx / frame_per_thread)].append(fno)


class Worker(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(maxsize=20)

    def decode(self, video_path, fnos, callback):
        self.queue.put((video_path, fnos, callback))

    def run(self):
        """the run loop to execute frame reading"""
        video_path, fnos, on_decode_callback = self.queue.get()
        cap = cv2.VideoCapture(video_path)

        # set initial frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnos[0])
        success = cap.grab()

        results = []
        idx, count = 0, fnos[0]
        while success:
            if count == fnos[idx]:
                success, image = cap.retrieve()
                if success:
                    on_decode_callback(image)
                else:
                    break
                idx += 1
                if idx >= len(fnos):
                    break
            count += 1
            success = cap.grab()


# queue for workers
worker_queue = queue.Queue()

# indicator for other threads to stop
stop_event = threading.Event()

# create and start threads
threads = []
for _ in range(0, n_threads):
    w = Worker()
    threads.append(w)
    w.start()

for idx, w in enumerate(threads):
    # w.decode(video_path, tasks[idx], on_done[idx])
    w.decode(video_path, tasks[idx], worker_queue, stop_event, idx)

# this will block until the first element is in the queue
first_finished = worker_queue.get()
print(f'Thread {first_finished} was first!')

# signal the rest to stop working
stop_event.set()

# results = queue.Queue(maxsize=100)
# results = [queue.Queue(maxsize=100) for i in range(n_threads)]
# on_done = lambda x: results.put(x)
# on_done = [lambda x,i=i: results[i].put(x) for i in range(n_threads)]

# distribute the tasks from main to worker threads

# do something with result now:
# t = Timer()
# t.start()
# min_ssim = 1
# while True:
#     img = results[0].get(timeout=5)
#     img_next = results[0].get(timeout=5)

#     height , width , layers =  img.shape
#     height_next , width_next , layers_next =  img_next.shape
#     img_new_w,img_next_new_w=int(width*.1),int(width_next*.1)
#     img_new_h,img_next_new_h=int(height*.1),int(height_next*.1)
#     dim = img_new_w,img_new_h
#     dim_next = img_next_new_w,img_next_new_h
#     resize = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#     resize_next = cv2.resize(img_next, dim_next, interpolation = cv2.INTER_AREA)
# #     current_ssim = frame_ssim_compare(resize,resize_next)
# #     if current_ssim < min_ssim:
# #         min_ssim = current_ssim
# #         print(min_ssim)
#     cv2.imshow('first',resize)
#     cv2.imshow('second',resize_next)
#     if cv2.waitKey(1) == ord('q'):
#         break
# t.stop()
