# to do -> handle mismatched aspect ratios and potention cropping, target search based on winning frame from consecutive ssim to priority search in taget ssim
# start brute force search within a window of original timestamp, then expand on each failure and avoid previously searched region
import csv
import os
from fractions import Fraction

import cv2
import ffmpeg
import numpy as np

import threaded_consecutive_ssim_threshold
from pprint import pprint

COMMON_FPS = 24000/1001


def get_fps(file):
    ffprobe_json = ffmpeg.probe(file, v='error', select_streams='v')
    fps = ffprobe_json['streams'][0]['r_frame_rate']
    fps_num, fps_den = fps.split('/')
    return int(fps_num), int(fps_den)


def iterate(func):
    def wrapper_iterate(*args, **kwargs):
        file_list_master = []
        paths = args
        for argument in args:
            file_list = []
            for root, dirs, files in os.walk(argument):
                try:
                    files.remove('.DS_Store')
                except:
                    pass
                for file in sorted(files):
                    file_list.append(os.path.join(root, file))
            file_list_master.append(file_list)
        for item in zip(*file_list_master):
            func(*item, **kwargs)
    return wrapper_iterate



def print_find_result(vid_1,vid_2):

    if vid_1.consecutive_frames_ssim is None:
        reference_vid = vid_1
        to_sync_vid = vid_2
    else:
        reference_vid = vid_2
        to_sync_vid = vid_1


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
            f'The local minimum SSIM between consecutive frames of {os.path.split(reference_vid.file)[1]} is {reference_vid.consecutive_frames_ssim}')
        if reference_vid.match_found:
            print(
                f'Match found between {os.path.split(reference_vid.file)[1]} and {os.path.split(to_sync_vid.file)[1]}!')
            print(
                f'The closest match has SSIM {to_sync_vid.match_frames_ssim} between F1s.')

            timestamp_difference = reference_vid.timestamp - to_sync_vid.timestamp
            if timestamp_difference > 0:
                print(
                    f'{os.path.split(reference_vid.file)[1]} is out of sync with {os.path.split(to_sync_vid.file)[1]}')
                print(f'V1F1 is number {reference_vid.timestamp}')
                print(f'V2F1 is number {to_sync_vid.timestamp}')
                print(
                    f'{os.path.split(to_sync_vid.file)[1]} needs to be delayed {timestamp_difference / 1000} seconds.\n')
            elif timestamp_difference < 0:
                print(
                    f'{os.path.split(reference_vid.file)[1]} is out of sync with {os.path.split(to_sync_vid.file)[1]}')
                print(f'V1F1 is number {reference_vid.timestamp}')
                print(f'V2F1 is number {to_sync_vid.timestamp}')
                print(
                    f'{os.path.split(reference_vid.file)[1]} needs to be delayed {-1*timestamp_difference / 1000} seconds.\n')
            else:
                print(
                    f'{os.path.split(reference_vid.file)[1]} is in sync with {os.path.split(to_sync_vid.file)[1]}\n')
                print(f'V1F1 is number {reference_vid.timestamp}')
                print(f'V2F1 is number {to_sync_vid.timestamp}')

            csv_writer.writerow(
                [reference_vid.file, to_sync_vid.file, reference_vid.match_found, reference_vid.timestamp, to_sync_vid.timestamp, timestamp_difference / 1000])
        else:
            print(
                f'No frame match found between {os.path.split(reference_vid.file)[1]} and {os.path.split(to_sync_vid.file)[1]}')
            print(
                f'The closest match has SSIM {to_sync_vid.match_frames_ssim} between F1s.\n')
            csv_writer.writerow([reference_vid.file, to_sync_vid.file, reference_vid.match_found])

def get_frame_aspect_ratio(frame):

    height, width, _ = frame.shape
    return width, height, Fraction(width, height)


def frame_is_letterboxed(img, th=25):
    # https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
    y_nonzero, x_nonzero, _ = np.nonzero(img > th)
    crop_height_mask = slice(np.min(y_nonzero), np.max(y_nonzero)+1)
    crop_width_mask = slice(np.min(x_nonzero), np.max(x_nonzero)+1)

    if np.array_equal(img[crop_height_mask, crop_width_mask], img):
        letterboxed = False
        return letterboxed, slice(None), slice(None)
    else:
        letterboxed = True
        return letterboxed, crop_height_mask, crop_width_mask


def crop_frame(img, crop_height_mask, crop_width_mask):
    return img[crop_height_mask, crop_width_mask]


class VideoSource():
    def __init__(self, file) -> None:
        self.file = file
        self.letterbox = None
        self.crop_height_mask = slice(None)
        self.crop_width_mask = slice(None)
        self.width_scaled = None
        self.height_scaled = None
        self.scale = None
        self.consecutive_frames = []
        self.consecutive_frames_timestamp = None
        self.consecutive_frames_number = None
        self.consecutive_frames_ssim = None
        self.match_frames_ssim = None
        self.non_black_frame = None
        self.sync_offset = None
        self.match_found = None

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
    elif width_1*height_1 < width_2*height_2:
        width_scaled = width_1
        height_scaled = height_1
        vid_1.scale = False
        vid_2.scale = True
    else:
        width_scaled = width_1
        height_scaled = height_1
        vid_1.scale = False
        vid_2.scale = False

    vid_1.width_scaled = width_scaled
    vid_1.height_scaled = height_scaled
    vid_2.width_scaled = width_scaled
    vid_2.height_scaled = height_scaled


# def quick_match_check(vid_1,vid_2):
#     quick_match = False

#     if vid_1.letterbox is None:
#         vid_1.set_vid_info()
#     if vid_2.letterbox is None:
#         vid_2.set_vid_info()

#     cap = cv2.VideoCapture(file_2)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, v1f1_number + offset)
#     v2f1_success, v2f1 = cap.read()
#     v2f2_success, v2f2 = cap.read()
#     if v2f1_success and v2f2_success:

#         if v1f_letterbox:
#             _, v1f1 = crop_frame_letterbox(v1f1)
#             _, v1f2 = crop_frame_letterbox(v1f2)
#         if v2f_letterbox:
#             _, v2f1 = crop_frame_letterbox(v2f1)
#             _, v2f2 = crop_frame_letterbox(v2f2)

#         if video_to_scale == 0:
#             v1f1 = threaded_consecutive_ssim_threshold.resize_frame(v1f1,dim=(width_scaled,height_scaled))
#             v1f2 = threaded_consecutive_ssim_threshold.resize_frame(v1f2,dim=(width_scaled,height_scaled))
#         elif video_to_scale == 1:
#             v2f1 = threaded_consecutive_ssim_threshold.resize_frame(v2f1,dim=(width_scaled,height_scaled))
#             v2f2 = threaded_consecutive_ssim_threshold.resize_frame(v2f2,dim=(width_scaled,height_scaled))
#         print(width_scaled,height_scaled)
#         print(v1f1.shape)
#         print(v2f1.shape)
#         result = threaded_consecutive_ssim_threshold.check_frame_match(v1f1, v2f1)
#         result_next = threaded_consecutive_ssim_threshold.check_frame_match(v1f2, v2f2)

#         if result and result_next:
#             print(f'{file_1} is in sync with {file_2}, with an offset of {offset} frames!')
#             quick_match = True
#             temp.append((v1f1, v1f2))
#         else:
#             print(
#                 f'{file_1} is out of sync with {file_2}, with an offset of {offset} frames!')
#             print(result, result_next)
#             temp.append((v1f1, v1f2))

def find(file_1, file_2):
    vid_1 = VideoSource(file_1)
    vid_2 = VideoSource(file_2)

    vid_1.set_vid_info()
    vid_2.set_vid_info()

    compare_aspect_dim(vid_1, vid_2)

    if vid_1.scale:
        threaded_consecutive_ssim_threshold.threaded_ssim(vid_2,
                                                                                                                        threaded_consecutive_ssim_threshold.Worker_Transition, min)
        high_res_vid = vid_1
        low_res_vid = vid_2
        threaded_consecutive_ssim_threshold.threaded_ssim(
        high_res_vid, threaded_consecutive_ssim_threshold.Worker_Transition_Match, max, video_source_2=low_res_vid)
    else:
        threaded_consecutive_ssim_threshold.threaded_ssim(vid_1,
                                                                                                                        threaded_consecutive_ssim_threshold.Worker_Transition, min)
        high_res_vid = vid_2
        low_res_vid = vid_1
        threaded_consecutive_ssim_threshold.threaded_ssim(
        high_res_vid, threaded_consecutive_ssim_threshold.Worker_Transition_Match, max, video_source_2=low_res_vid)

    print_find_result(vid_1, vid_2)
    return vid_1,vid_2


def batch_find(dir_1,dir_2,view=None):
    if view is None:
        view = False
    list_1 = sorted((f for f in os.listdir(dir_1) if not f.startswith(".")), key=str.lower)
    list_2 = sorted((f for f in os.listdir(dir_2) if not f.startswith(".")), key=str.lower)

    for i,(file_1,file_2) in enumerate(zip(list_1,list_2)):
        v1f1, v1f2, v2f1, v2f2, v1f1_v1f2_min_ssim, v1f1_v2f1_ssim, v1f1_number, v2f1_number, v1f1_timestamp, v2f1_timestamp, match_found = find(file_1,file_2)

        if view:
            while True:
                cv2.imshow('V1F1', v1f1)
                cv2.imshow('V1F2', v1f2)
                cv2.imshow('V2F1', v2f1)
                cv2.imshow('V2F2', v2f2)
                if cv2.waitKey(1) == ord('q'):
                    break

find()
