import math
from fractions import Fraction

import cv2
import ffmpeg
import numpy as np
from skimage.metrics import structural_similarity as ssim

COMMON_FPS = 24000/1001
MAX_SSIM = 1
MIN_SSIM = -1
SSIM_SCORE_THRESHOLD = .2


def consecutive_ssim_metric(frame_1, frame_2):
    return math.sqrt(abs(frame_1*frame_2))


def frame_colored(frame):
    # Goal is to avoid mostly all black or white frames. Arbitrary cutoff: 0 is black, 255 is white
    return 30 < np.mean(frame) < 230


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


def get_frame_aspect_ratio(frame):
    height, width, _ = frame.shape
    return width, height, Fraction(width, height)


def get_fps(file):
    ffprobe_json = ffmpeg.probe(file, v='error', select_streams='v')
    fps = ffprobe_json['streams'][0]['r_frame_rate']
    fps_num, fps_den = fps.split('/')
    return int(fps_num), int(fps_den)


def get_fps_cv_native(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def vfr_cfr_check(video_path):
    # Need to rethink this...
    # https://forum.videohelp.com/threads/379834-constant-variable-frame-rate-confusion-again
    # https://sourceforge.net/p/mediainfo/discussion/297610/thread/b2a2708d/
    # https://superuser.com/questions/1487401/how-can-i-tell-if-a-video-has-a-variable-frame-rate
    # 24000/1001 fps results in 41,42 ms differences between frames
    # might be worth just checking with something like MediaInfo
    timestamps = []
    cap = cv2.VideoCapture(video_path)
    for i in range(10):
        success, frame = cap.read()
        timestamp = round(cap.get(cv2.CAP_PROP_POS_MSEC))
        timestamps.append(timestamp)
    differences = [timestamps[i]-timestamps[i-1]
                   for i in range(1, len(timestamps))]
    cap.release()

    if all(x == differences[0] for x in differences):
        return differences, "Constant"
    else:
        return differences, "Variable"


def view_frames(*args):
    # Solution: https://stackoverflow.com/questions/52682261/why-cv2-destroyallwindows-does-not-work-on-mac
    for i, arg in enumerate(args):
        cv2.imshow(f'Frame {i}', arg)
    cv2.startWindowThread()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for i in range(2):
        cv2.waitKey(1)


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


def crop_frame(img, crop_height_mask, crop_width_mask):
    return img[crop_height_mask, crop_width_mask]


def transform_frame(frame, scaling_factor=.1, dim=None, cropped=None):
    if cropped is not None:
        frame = crop_frame(frame, *cropped[1:])
    if dim is not None:
        frame = resize_frame(frame, dim=dim)
    scaled_down_frame = resize_frame(frame, scaling_factor, scaling_factor)
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


def ssim_iterate_transition_frames(video_path):
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
    ssim_scores = ssim_iterate_transition_frames(video_path)
    min_index = min(ssim_scores, key=ssim_scores.get)
    if view:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min_index)
        _, frame_1 = cap.read()
        _, frame_2 = cap.read()
        view_frames(frame_1, frame_2)
    return min_index
