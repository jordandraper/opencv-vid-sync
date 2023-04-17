import csv
import os

import cv2
import numpy as np

import workers
from helpers import frame_is_letterboxed,crop_frame,get_frame_aspect_ratio,view_frames

def print_find_result(vid_1, vid_2):
    if vid_1.transition_frames_ssim is None:
        reference_vid = vid_2
        to_sync_vid = vid_1
    else:
        reference_vid = vid_1
        to_sync_vid = vid_2

    if not os.path.isfile(os.path.join(os.path.split(vid_1.file)[0], 'frame_sync.csv')):
        with open(os.path.join(os.path.split(vid_2.file)[0], 'frame_sync.csv'), mode='w') as csvfile:
            csv_writer = csv.writer(
                csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(
                ["File 1", "File 2", "Match Found", "V1F1 Number", "V2F1 Number", "Delay"])

    with open(os.path.join(os.path.split(vid_1.file)[0], 'frame_sync.csv'), mode='a') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        print(
            f'The local minimum SSIM between consecutive frames of {os.path.split(reference_vid.file)[1]} is {reference_vid.transition_frames_ssim}')
        if reference_vid.match_found:
            print(
                f'Match found between {os.path.split(reference_vid.file)[1]} and {os.path.split(to_sync_vid.file)[1]}!')
            print(
                f'The closest match has SSIM {to_sync_vid.match_frames_ssim} between frames.')

            timestamp_difference = reference_vid.transition_frames_timestamp - \
                to_sync_vid.transition_frames_timestamp
            if timestamp_difference > 0:
                print(
                    f'{os.path.split(reference_vid.file)[1]} is out of sync with {os.path.split(to_sync_vid.file)[1]}')
                print(
                    f'V1F1 timestamp is {reference_vid.transition_frames_timestamp}')
                print(
                    f'V2F1 timestamp is {to_sync_vid.transition_frames_timestamp}')
                print(
                    f'{os.path.split(to_sync_vid.file)[1]} needs to be delayed {timestamp_difference / 1000} seconds.\n')
            elif timestamp_difference < 0:
                print(
                    f'{os.path.split(reference_vid.file)[1]} is out of sync with {os.path.split(to_sync_vid.file)[1]}')
                print(
                    f'V1F1 timestamp is {reference_vid.transition_frames_timestamp}')
                print(
                    f'V2F1 timestamp is {to_sync_vid.transition_frames_timestamp}')
                print(
                    f'{os.path.split(reference_vid.file)[1]} needs to be delayed {-1*timestamp_difference / 1000} seconds.\n')
            else:
                print(
                    f'{os.path.split(reference_vid.file)[1]} is in sync with {os.path.split(to_sync_vid.file)[1]}\n')
                print(
                    f'V1F1 timestamp is {reference_vid.transition_frames_timestamp}')
                print(
                    f'V2F1 timestamp is {to_sync_vid.transition_frames_timestamp}')

            csv_writer.writerow(
                [reference_vid.file, to_sync_vid.file, reference_vid.match_found, reference_vid.transition_frames_timestamp, to_sync_vid.transition_frames_timestamp, timestamp_difference / 1000])
        else:
            print(
                f'No frame match found between {os.path.split(reference_vid.file)[1]} and {os.path.split(to_sync_vid.file)[1]}')
            print(
                f'The closest match has SSIM {to_sync_vid.match_frames_ssim} between frames.\n')
            csv_writer.writerow(
                [reference_vid.file, to_sync_vid.file, reference_vid.match_found])


class VideoSource():
    def __init__(self, file) -> None:
        self.file = file
        self.letterbox = None
        self.crop_height_mask = slice(None)
        self.crop_width_mask = slice(None)
        self.width_scaled = None
        self.height_scaled = None
        self.scale = None
        self.transition_frames = None
        self.transition_frames_timestamp = None
        self.transition_frames_number = None
        self.transition_frames_ssim = None
        self.match_frames_ssim = None
        self.non_black_frame = None
        self.sync_offset = None
        self.match_found = None
        self.reference_vid = None
        self.to_sync_vid = None

    def set_vid_info(self):
        """Get first non-black frame and check for letterboxing"""
        cap_1 = cv2.VideoCapture(self.file)
        while True:
            _, v1f = cap_1.read()

            # converts the frame to gray scale for easier computation
            gray = cv2.cvtColor(v1f, cv2.COLOR_BGR2GRAY)

            if np.average(gray) < 90:
                # skips an iteration, so the frame isn't saved
                continue
            else:
                break
        self.non_black_frame = v1f
        self.letterbox, self.crop_height_mask, self.crop_width_mask = frame_is_letterboxed(
            v1f)


def compare_aspect_dim(vid_1, vid_2):
    cropped_v1f = crop_frame(vid_1.non_black_frame,
                             vid_1.crop_height_mask, vid_1.crop_width_mask)
    cropped_v2f = crop_frame(vid_2.non_black_frame,
                             vid_2.crop_height_mask, vid_2.crop_width_mask)

    width_1, height_1, aspect_1 = get_frame_aspect_ratio(cropped_v1f)
    width_2, height_2, aspect_2 = get_frame_aspect_ratio(cropped_v2f)

    if width_1*height_1 > width_2*height_2:
        width_scaled = width_2
        height_scaled = height_2
        vid_1.scale = True
        vid_2.scale = False
        vid_2.reference_vid = True
        vid_2.to_sync_vid = False
        vid_1.reference_vid = False
        vid_1.to_sync_vid = True
    elif width_1*height_1 < width_2*height_2:
        width_scaled = width_1
        height_scaled = height_1
        vid_1.scale = False
        vid_2.scale = True
        vid_2.reference_vid = False
        vid_2.to_sync_vid = True
        vid_1.reference_vid = True
        vid_1.to_sync_vid = False
    else:
        width_scaled = width_1
        height_scaled = height_1
        vid_1.scale = False
        vid_2.scale = False
        vid_2.reference_vid = False
        vid_2.to_sync_vid = True
        vid_1.reference_vid = True
        vid_1.to_sync_vid = False

    vid_1.width_scaled = width_scaled
    vid_1.height_scaled = height_scaled
    vid_2.width_scaled = width_scaled
    vid_2.height_scaled = height_scaled

def find(file_1, file_2):
    vid_1 = VideoSource(file_1)
    vid_2 = VideoSource(file_2)

    vid_1.set_vid_info()
    vid_2.set_vid_info()

    compare_aspect_dim(vid_1, vid_2)

    if vid_2.reference_vid:
        reference_vid = vid_2
        to_sync_vid = vid_1
    else:
        reference_vid = vid_1
        to_sync_vid = vid_2

    workers.threaded_ssim(reference_vid,
                          workers.Worker_Transition, min, frame_window=[0, 100])

    quick_match = workers.quick_match_check(
        reference_vid, to_sync_vid)

    if not quick_match:
        workers.threaded_ssim(
            to_sync_vid, workers.Worker_Transition_Match, max, video_source_2=reference_vid, frame_window=[0, 300])
        print_find_result(vid_1, vid_2)
    return vid_1, vid_2


def batch_find(dir_1, dir_2, view=None):
    if view is None:
        view = False
    list_1 = sorted((f for f in os.listdir(dir_1)
                    if not f.startswith(".")), key=str.lower)
    list_2 = sorted((f for f in os.listdir(dir_2)
                    if not f.startswith(".")), key=str.lower)

    for i, (file_1, file_2) in enumerate(zip(list_1, list_2)):
        vid_1,vid_2 = find(
            file_1, file_2)

        if view:
            view_frames(vid_1.transition_frames[0],vid_2.transition_frames[0])
            view_frames(vid_1.transition_frames[1],vid_2.transition_frames[1])
