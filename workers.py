import math
import queue
import threading

import cv2
from skimage.metrics import structural_similarity as ssim

import timer
from helpers import (MAX_SSIM, MIN_SSIM, SSIM_SCORE_THRESHOLD, transform_frame,
                     view_frames)


def quick_match_check(reference_vid, to_sync_vid, frame_offset=None, time_offset=None):
    quick_match = False

    target_frame_1, target_frame_2 = reference_vid.transition_frames
    target_frame_1 = transform_frame(target_frame_1, dim=(
        reference_vid.width_scaled, reference_vid.height_scaled), cropped=(reference_vid.letterbox, reference_vid.crop_height_mask, reference_vid.crop_width_mask))
    target_frame_2 = transform_frame(target_frame_2, dim=(
        reference_vid.width_scaled, reference_vid.height_scaled), cropped=(reference_vid.letterbox, reference_vid.crop_height_mask, reference_vid.crop_width_mask))

    cap = cv2.VideoCapture(to_sync_vid.file)

    if frame_offset is not None:
        # set position by frame
        cap.set(cv2.CAP_PROP_POS_FRAMES,
                reference_vid.transition_frames_number + frame_offset)
        positive_output = f'{reference_vid.file} is in sync with {to_sync_vid.file} with an offset of {frame_offset} frames!'
        negative_output = f'{reference_vid.file} is out of sync with {to_sync_vid.file} with an offset of {frame_offset} frames!'

    elif time_offset is not None:
        # set position by timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC,
                reference_vid.transition_frames_timestamp + time_offset)
        positive_output = f'{reference_vid.file} is in sync with {to_sync_vid.file} with an offset of {time_offset/1000} milliseconds!'
        negative_output = f'{reference_vid.file} is out of sync with {to_sync_vid.file} with an offset of {time_offset/1000} milliseconds!'
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, reference_vid.transition_frames_timestamp)
        positive_output = f'{reference_vid.file} is in sync with {to_sync_vid.file}!'
        negative_output = f'{reference_vid.file} is out of sync with {to_sync_vid.file}!'
    _, frame_1 = cap.read()
    success, frame_2 = cap.read()
    if success:
        if to_sync_vid.scale:
            resized_frame_1 = transform_frame(frame_1, dim=(
                to_sync_vid.width_scaled, to_sync_vid.height_scaled), cropped=(to_sync_vid.letterbox, to_sync_vid.crop_height_mask, to_sync_vid.crop_width_mask))
            resized_frame_2 = transform_frame(frame_2, dim=(
                to_sync_vid.width_scaled, to_sync_vid.height_scaled), cropped=(to_sync_vid.letterbox, to_sync_vid.crop_height_mask, to_sync_vid.crop_width_mask))
        else:
            resized_frame_1 = transform_frame(
                frame_1, cropped=(to_sync_vid.letterbox, to_sync_vid.crop_height_mask, to_sync_vid.crop_width_mask))
            resized_frame_2 = transform_frame(
                frame_2, cropped=(to_sync_vid.letterbox, to_sync_vid.crop_height_mask, to_sync_vid.crop_width_mask))
        ssim_score_target_frame_1 = ssim(
            resized_frame_1, target_frame_1)
        ssim_score_target_frame_2 = ssim(
            resized_frame_2, target_frame_2)
        if ssim_score_target_frame_1 > .9 and ssim_score_target_frame_2 > .9:
            print(positive_output)
            quick_match = True
            reference_vid.match_found = True
            to_sync_vid.match_found = True
        else:
            print(negative_output)
            view = input(
                f"The quick match SSIM is {ssim_score_target_frame_1}. Would you like to manually view?")
            if view.lower() == "yes" or view.lower() == 'y':
                view_frames(resized_frame_1, target_frame_1)
                manual_inspect = input(f"Are the frames in sync?")
                if manual_inspect.lower() == "yes" or manual_inspect.lower() == 'y':
                    quick_match = True
                    reference_vid.match_found = True
                    to_sync_vid.match_found = True
    return quick_match


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
        success, frame_1 = cap.read()
        if success:
            while True and frame_number <= fnos[-1]:
                if stop_event.is_set():
                    break
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                success, frame_2 = cap.read()
                if success:
                    resized_frame_1 = transform_frame(frame_1)
                    resized_frame_2 = transform_frame(frame_2)
                    ssim_score = ssim(resized_frame_1, resized_frame_2)

                    if ssim_score < min_ssim:
                        min_ssim = ssim_score
                        min_frame = {"Transition Frame SSIM": min_ssim, "Transition Frame Number": frame_number,
                                     "Transition Frame Timestamp": timestamp, "Transition Frames": [frame_1, frame_2]}

                    # avoid completely black frames
                    if ssim_score < SSIM_SCORE_THRESHOLD and cv2.countNonZero(resized_frame_1) and cv2.countNonZero(resized_frame_2):
                        stop_event.set()
                        result.update(min_frame)
                        worker_queue.put(self)
                        cap.release()
                        return

                    frame_number += 1
                    frame_1 = frame_2
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

        target_frame_1, target_frame_2 = video_source_2.transition_frames
        target_frame_1 = transform_frame(target_frame_1, dim=(
            video_source_2.width_scaled, video_source_2.height_scaled), cropped=(video_source_2.letterbox, video_source_2.crop_height_mask, video_source_2.crop_width_mask))
        target_frame_2 = transform_frame(target_frame_2, dim=(
            video_source_2.width_scaled, video_source_2.height_scaled), cropped=(video_source_2.letterbox, video_source_2.crop_height_mask, video_source_2.crop_width_mask))

        frame_number = fnos[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame_1 = cap.read()
        if success:
            while True and frame_number <= fnos[-1]:
                if stop_event.is_set():
                    break
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                success, frame_2 = cap.read()
                if success:
                    if video_source_1.scale:
                        resized_frame_1 = transform_frame(frame_1, dim=(
                            video_source_1.width_scaled, video_source_1.height_scaled), cropped=(video_source_1.letterbox, video_source_1.crop_height_mask, video_source_1.crop_width_mask))
                        resized_frame_2 = transform_frame(frame_2, dim=(
                            video_source_1.width_scaled, video_source_1.height_scaled), cropped=(video_source_1.letterbox, video_source_1.crop_height_mask, video_source_1.crop_width_mask))
                    else:
                        resized_frame_1 = transform_frame(
                            frame_1, cropped=(video_source_1.letterbox, video_source_1.crop_height_mask, video_source_1.crop_width_mask))
                        resized_frame_2 = transform_frame(
                            frame_2, cropped=(video_source_1.letterbox, video_source_1.crop_height_mask, video_source_1.crop_width_mask))

                    ssim_score_target_frame_1 = ssim(
                        resized_frame_1, target_frame_1)
                    ssim_score_target_frame_2 = ssim(
                        resized_frame_2, target_frame_2)

                    if ssim_score_target_frame_1 > max_ssim:
                        max_ssim = ssim_score_target_frame_1
                        max_frame = {"Match Frame SSIM": max_ssim, "Transition Frame Number": frame_number,
                                     "Transition Frame Timestamp": timestamp, "Transition Frames": [frame_1, frame_2]}

                    if ssim_score_target_frame_1 > .9 and ssim_score_target_frame_2 > .9:
                        stop_event.set()
                        max_frame["Match Found"] = True
                        result.update(max_frame)
                        worker_queue.put(self)
                        cap.release()
                        return
                    frame_number += 1
                    frame_1 = frame_2

            if not stop_event.is_set():
                max_frame["Match Found"] = False
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
                                 ssim_score_target_frame, frame_number, timestamp]
                    max_ssim = ssim_score_target_frame

                if ssim_score_target_frame > .9:
                    stop_event.set()
                    worker_queue.put(self)
                    result.extend(max_frame)
                    cap.release()
                    return
                frame_number += 1
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
        total_frames = frame_window[1]-frame_window[0]+1
        fnos = list(range(frame_window[0], frame_window[1]+1))

    n_threads = 4  # n_threads is the number of worker threads to read video frame
    # store frame number for each thread
    tasks = [[] for _ in range(n_threads)]
    frames_per_thread = math.ceil(total_frames / n_threads)

    for idx, fno in enumerate(fnos):
        tasks[math.floor(idx / frames_per_thread)].append(fno)

    # structure to hold winner and no matches
    result = {"Transition Frame SSIM": None, "Transition Frame Number": None,
              "Transition Frame Timestamp": None, "Transition Frames": None, "Match Frame SSIM": None, "Match Found": None}
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

    t = timer.Timer()
    t.start()
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
            if closest_match is max:
                closest_frame = closest_match(
                    no_match, key=lambda x: x["Match Frame SSIM"])
            elif closest_match is min:
                closest_frame = closest_match(
                    no_match, key=lambda x: x["Transition Frame SSIM"])
            result.update(closest_frame)
            break

    if video_source_2 is None:
        video_source_1.transition_frames_ssim = result.get(
            "Transition Frame SSIM")
    else:
        video_source_1.match_frames_ssim = result.get("Match Frame SSIM")
        video_source_1.match_found = result.get("Match Found")

        if video_source_1.match_found:
            video_source_2.match_found = True
        else:
            video_source_2.match_found = False
    video_source_1.transition_frames_number = result.get(
        "Transition Frame Number")
    video_source_1.transition_frames_timestamp = result.get(
        "Transition Frame Timestamp")
    video_source_1.transition_frames = result.get("Transition Frames")

    t.stop()
    # this will block until the first element is in the queue
    # worker_queue.get()

    cap.release()
    return result
