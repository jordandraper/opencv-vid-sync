# Thoughts: while this is good for retrieving frames, it does not do so sequentially because threads return frames to the same queue and therefore is not useful for comparing SSIM between sequential frames
import math
import queue
import threading

import cv2
from skimage.metrics import structural_similarity as ssim

from threaded_consecutive_ssim_threshold import frame_ssim_compare


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

        idx, count = 0, fnos[0]
        while success:
            if count == fnos[idx]:
                success, img = cap.retrieve()
                height, width, _ = img.shape
                dim = int(width*.1), int(height*.1)
                resize = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                resize = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
                if success:
                    on_decode_callback(resize)
                else:
                    break
                idx += 1
                if idx >= len(fnos):
                    break
            count += 1
            success = cap.grab()


def threaded_target_ssim_threshold(video_path, target_frame):

    cap = cv2.VideoCapture(video_path)
    sample_rate = 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fnos = list(range(0, total_frames, sample_rate))
    n_threads = 4  # n_threads is the number of worker threads to read video frame
    # store frame number for each threads
    tasks = [[] for _ in range(0, n_threads)]
    frames_per_thread = math.ceil(len(fnos) / n_threads)

    for idx, fno in enumerate(fnos):
        tasks[math.floor(idx / frames_per_thread)].append(fno)

    # create and start threads
    threads = []
    for _ in range(0, n_threads):
        w = Worker()
        # daemon threads will die once ssim threshold is reached since program ends
        w.setDaemon(True)
        threads.append(w)
        w.start()

    results = queue.Queue(maxsize=100)
    def on_done(x): return results.put(x)
    # distribute the tasks from main to worker threads
    for idx, w in enumerate(threads):
        w.decode(video_path, tasks[idx], on_done)

    # do something with result now:
    max_ssim = -1
    while True:
        try:
            img = results.get(timeout=5)
            s = ssim(img, target_frame)
            if s > max_ssim:
                max_ssim = s
            if max_ssim >= .97:
                return img, target_frame, max_ssim
        except queue.Empty:
            return None, target_frame, max_ssim
