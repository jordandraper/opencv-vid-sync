import math
import queue
import threading
from datetime import timedelta

import cv2
from skimage.metrics import structural_similarity as ssim

MAX_SSIM = 1
MIN_SSIM = -1
SSIM_SCORE_THRESHOLD = .2


def crop_frame_letterbox(img):
    # Read the image, convert it into grayscale, and make in binary image for threshold value of 1.
    # img = cv2.imread('sofwin.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Now find contours in it. There will be only one object, so find bounding rectangle for it.
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)

        # Now crop the image, and save it into another file.
        crop = img[y:y+h, x:x+w]
        # view_frames(img,crop)
        # cv2.imwrite('sofwinres.png',crop)
        return True, crop
    else:
        return False, img


def view_frame(frame):
    while True:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break


def view_frames(frame_1, frame_2):
    while True:
        cv2.imshow('Frame 1', frame_1)
        cv2.imshow('Frame 2', frame_2)
        if cv2.waitKey(1) == ord('q'):
            break


def resize_frame(frame, width_scaling_factor=None, height_scaling_factor=None, dim=None):
    if width_scaling_factor is None:
        width_scaling_factor = 1
    if height_scaling_factor is None:
        height_scaling_factor = 1
    if dim is None:
        height, width, _ = frame.shape
        dim = int(width*width_scaling_factor), int(height *
                                                   height_scaling_factor)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def transform_frame(frame, scaling_factor=.1, dim=None, cropped=False):
    # print('OG Frame:')
    # print(frame.shape)
    # if cropped:
    #     _, frame = crop_frame_letterbox(frame)
    #     print('Just cropped:')
    #     print(frame.shape)
    if dim is not None:
        frame = resize_frame(frame, dim=dim)
    scaled_down_frame = resize_frame(frame, scaling_factor, scaling_factor)

    # print("Resized frame:")
    # print(scaled_down_frame.shape)
    grey_scale_frame = cv2.cvtColor(scaled_down_frame, cv2.COLOR_BGR2GRAY)
    return grey_scale_frame


def calculate_frame_ssim(frame_1, frame_2):
    resized_frame_1 = transform_frame(frame_1)
    resized_frame_2 = transform_frame(frame_2)
    return ssim(resized_frame_1, resized_frame_2)


def check_frame_match(frame_1, frame_2, match_threshold=None):
    if match_threshold is None:
        match_threshold = .9
    ssim_score = calculate_frame_ssim(frame_1, frame_2)
    if ssim_score >= match_threshold:
        return True
    else:
        return False


def ssim_iterate_consecutive_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    idx = 0
    ssim_scores = {}
    _, frame_1 = cap.read()
    while True:
        success, frame_2 = cap.read()
        if success:
            ssim_score = calculate_frame_ssim(frame_1, frame_2)
            ssim_scores[idx] = ssim_score
            idx += 1
            frame_1 = frame_2
        else:
            break
    return ssim_scores


def find_transition(video_path, view=None):
    if view is None:
        view = False
    ssim_scores = ssim_iterate_consecutive_frames(video_path)
    min_index = min(ssim_scores, key=ssim_scores.get)
    if view:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min_index)
        _, frame_1 = cap.read()
        _, frame_2 = cap.read()
        view_frames(frame_1, frame_2)
    return min_index


class Worker_Transition(threading.Thread):
    """
    Find a scene transition to use as reference point. Transition is based on SSIM_SCORE_THRESHOLD.
    """

    def __init__(self):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(maxsize=20)

    def decode(self, video_source, fnos, worker_queue, stop_event, result, no_match):
        self.queue.put((video_source, fnos, worker_queue, stop_event,
                        result, no_match))

    def run(self):
        video_source, fnos, worker_queue, stop_event, result, no_match = self.queue.get()

        cap = cv2.VideoCapture(video_source.file)
        min_ssim = MAX_SSIM

        frame_number = fnos[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        while True:
            if stop_event.is_set():
                break
            _, frame_1 = cap.read()
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            success, frame_2 = cap.read()
            if success:
                resized_frame_1 = transform_frame(frame_1)
                resized_frame_2 = transform_frame(frame_2)
                ssim_score = ssim(resized_frame_1, resized_frame_2)

                if ssim_score <= min_ssim:
                    min_frame = [frame_1, frame_2,
                                 ssim_score, frame_number, timestamp]
                    min_ssim = ssim_score

                    video_source.consecutive_frames_ssim = min_ssim
                    video_source.consecutive_frames_number = frame_number
                    video_source.consecutive_frames_timestamp = timestamp

            frame_number += 1

            # avoid completely black frames
            if ssim_score < SSIM_SCORE_THRESHOLD and cv2.countNonZero(resized_frame_1) and cv2.countNonZero(resized_frame_2):
                stop_event.set()
                result.extend(min_frame)
                worker_queue.put(self)
                video_source.consecutive_frames.extend([frame_1, frame_2])
                return
        if not stop_event.is_set():
            no_match.append(min_frame)
            cap.release()


class Worker_Transition_Match(threading.Thread):
    """
    Find two consecutive frames which match the target scene change with joint SSIM greater than 90/100.
    """

    def __init__(self):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(maxsize=20)

    def decode(self, video_source_1, fnos, worker_queue, stop_event, result, no_match, video_source_2):
        self.queue.put((video_source_1, fnos, worker_queue,
                       stop_event, result, no_match, video_source_2))

    def run(self):
        video_source_1, fnos, worker_queue, stop_event, result, no_match, video_source_2 = self.queue.get()
        cap = cv2.VideoCapture(video_source_1.file)
        max_ssim = MIN_SSIM

        target_frame_1, target_frame_2 = video_source_2.consecutive_frames
        target_frame_1 = transform_frame(target_frame_1, dim=(
            video_source_2.width_scaled, video_source_2.height_scaled), cropped=video_source_2.letterbox)
        target_frame_2 = transform_frame(target_frame_2, dim=(
            video_source_2.width_scaled, video_source_2.height_scaled), cropped=video_source_2.letterbox)

        frame_number = fnos[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        while True:
            if stop_event.is_set():
                break
            _, frame_1 = cap.read()
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            success, frame_2 = cap.read()
            if success:
                if video_source_1.scale:
                    resized_frame_1 = transform_frame(frame_1, dim=(
                        video_source_1.width_scaled, video_source_1.height_scaled), cropped=video_source_1.letterbox)
                    resized_frame_2 = transform_frame(frame_2, dim=(
                        video_source_1.width_scaled, video_source_1.height_scaled), cropped=video_source_1.letterbox)
                else:
                    resized_frame_1 = transform_frame(
                        frame_1, cropped=video_source_1.letterbox)
                    resized_frame_2 = transform_frame(
                        frame_2, cropped=video_source_1.letterbox)

                ssim_score_target_frame_1 = ssim(
                    resized_frame_1, target_frame_1)
                ssim_score_target_frame_2 = ssim(
                    resized_frame_2, target_frame_2)

                if ssim_score_target_frame_1 >= max_ssim:
                    max_frame = [frame_1, frame_2,
                                 ssim_score_target_frame_1, frame_number, timestamp]
                    max_ssim = ssim_score_target_frame_1

                    video_source_1.match_frames_ssim = max_ssim
                    video_source_1.consecutive_frames_number = frame_number
                    video_source_1.consecutive_frames_timestamp = timestamp
                    video_source_1.consecutive_frames.extend([frame_1, frame_2])

                frame_number += 1

                if ssim_score_target_frame_1 > .9 and ssim_score_target_frame_2 > .9:
                    stop_event.set()
                    worker_queue.put(self)
                    max_frame.extend([True])
                    result.extend(max_frame)
                    video_source_1.match_found = True
                    video_source_2.match_found = True
                    return
        if not stop_event.is_set():
            video_source_1.match_found = False
            video_source_2.match_found = False
            max_frame.extend([False])
            no_match.append(max_frame)
            cap.release()


class Worker_Frame_Match(threading.Thread):
    """
    Linearly search for a single frame match.
    """

    def __init__(self):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(maxsize=20)

    def decode(self, video_path, fnos, worker_queue, stop_event, result, no_match, target_frame):
        self.queue.put((video_path, fnos, worker_queue,
                       stop_event, result, no_match, target_frame))

    def run(self):
        video_path, fnos, worker_queue, stop_event, result, no_match, target_frame = self.queue.get()
        cap = cv2.VideoCapture(video_path)
        max_ssim = MIN_SSIM

        frame_number = fnos[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        while True:
            if stop_event.is_set():
                break
            success, frame_1 = cap.read()
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            if success:
                resized_frame_1 = transform_frame(frame_1)
                ssim_score_target_frame = ssim(resized_frame_1, target_frame)

                if ssim_score_target_frame >= max_ssim:
                    max_frame = [frame_1, target_frame,
                                 ssim_score_target_frame, fno, timestamp]
                    max_ssim = ssim_score_target_frame

                frame_number += 1

                if ssim_score_target_frame > .9:
                    stop_event.set()
                    worker_queue.put(self)
                    result.extend(max_frame)
                    return
        if not stop_event.is_set():
            no_match.append(max_frame)
            cap.release()


def threaded_ssim(video_source_1, worker, closest_match, video_source_2=None, frame_window=None):
    """
    closest_match is preference for max/min when there is no match found for all threads
    """

    cap = cv2.VideoCapture(video_source_1.file)

    if frame_window is None:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fnos = list(range(total_frames))
    else:
        total_frames = int(frame_window[1]-frame_window[0])
        fnos = list(range(frame_window[0], frame_window[1]+1))

    n_threads = 4  # n_threads is the number of worker threads to read video frame
    # store frame number for each threads
    tasks = [[] for _ in range(n_threads)]
    frames_per_thread = math.ceil(total_frames / n_threads)

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
    for _ in range(n_threads):
        w = worker()
        threads.append(w)
        w.start()

    if video_source_2 is not None:
        for idx, w in enumerate(threads):
            w.decode(video_source_1,
                     tasks[idx], worker_queue, stop_event, result, no_match, video_source_2)
    else:
        for idx, w in enumerate(threads):
            w.decode(video_source_1,
                     tasks[idx], worker_queue, stop_event, result, no_match)

    while worker_queue.empty():
        if len(no_match) != n_threads:
            pass
        else:
            worker_queue.put('No match!')
            closest_frame = closest_match(no_match, key=lambda x: x[2])
            result.extend(closest_frame)
            break

    # this will block until the first element is in the queue
    first_finished = worker_queue.get()

    # print(f'Thread {first_finished} was first!')
    # print(result)
    # td = timedelta(seconds=result[4]/1000)
    # print(f'SSIM score of {result[2]}, Frame number {result[3]}, Time: {td}')

    cap.release()
    return result
