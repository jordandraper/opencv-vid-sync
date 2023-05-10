import csv
import os
import sys

import cv2
import numpy as np
import workers
from helpers import (concat_frames, crop_frame, frame_is_letterboxed,
                     get_fps_cv_native, get_frame_aspect_ratio, view_frame)


def print_find_result(vid_1, vid_2):
    if vid_1.reference_vid:
        reference_vid = vid_1
        to_sync_vid = vid_2
    else:
        reference_vid = vid_2
        to_sync_vid = vid_1

    if not os.path.isfile(os.path.join(os.path.dirname(sys.argv[0]), "results", 'frame_sync.csv')):
        with open(os.path.join(os.path.dirname(sys.argv[0]), "results", 'frame_sync.csv'), mode='w') as csvfile:
            csv_writer = csv.writer(
                csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(
                ["File 1", "File 2", "Match Found", "V1F1 Timestamp", "V2F1 Timestamp", "Delay"])

    with open(os.path.join(os.path.dirname(sys.argv[0]), "results", 'frame_sync.csv'), mode='a') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        timestamp_difference = reference_vid.transition_frames_timestamp - \
            to_sync_vid.transition_frames_timestamp
        if reference_vid.match_found:
            print(
                f'\nMatch found between {reference_vid.name} and {to_sync_vid.name}!')
            print(
                f'The closest match has SSIM {to_sync_vid.match_frames_ssim} between frames.\n')
            print(
                f'{reference_vid.name} timestamp is {reference_vid.transition_frames_timestamp}ms.')
            print(
                f'{to_sync_vid.name} timestamp is {to_sync_vid.transition_frames_timestamp}ms.')
            print(
                f'{to_sync_vid.name} needs to be delayed {timestamp_difference} milliseconds.\n')

            csv_writer.writerow(
                [reference_vid.file, to_sync_vid.file, reference_vid.match_found, reference_vid.transition_frames_timestamp, to_sync_vid.transition_frames_timestamp, timestamp_difference])
        else:
            print(
                f'\nNo frame match found between {reference_vid.name} and {to_sync_vid.name}')
            print(
                f'The closest match has SSIM {to_sync_vid.match_frames_ssim} between frames.\n')
            view = input(
                f"Would you like to manually view? ")
            if view.lower() == "yes" or view.lower() == 'y':
                view_frame(
                    concat_frames((reference_vid.transition_frames[0], to_sync_vid.transition_frames[0])))
                manual_inspect = input(f"Are the frames in sync? ")
                if manual_inspect.lower() == "yes" or manual_inspect.lower() == 'y':
                    reference_vid.match_found = True
                    to_sync_vid.match_found = True

                    csv_writer.writerow(
                        [reference_vid.file, to_sync_vid.file, reference_vid.match_found, reference_vid.transition_frames_timestamp, to_sync_vid.transition_frames_timestamp, timestamp_difference])
                else:
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
        self.fps = None
        self.name = os.path.split(file)[1]
        # self.framerate_type = None
        # self.frame_difference_sample = None

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
        # fps can be tricky e.g. 24000/1001 vs 23976/1000 vs 2997/125. Both round to 23.976
        self.fps = round(get_fps_cv_native(self.file), 3)
        # self.frame_difference_sample, self.framerate_type = vfr_cfr_check(self.file)


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


def find(file_1, file_2, view=None, save_frames=None):
    if view is None:
        view = False
    if save_frames is None:
        save_frames = False

    vid_1 = VideoSource(file_1)
    vid_2 = VideoSource(file_2)

    vid_1.set_vid_info()
    vid_2.set_vid_info()

    # if vid_1.framerate_type == "Variable" or vid_2.framerate_type == "Variable":
    #     print("Input video has variable framerate. Sync might not possible.")
    #     sample = input("Would you like to compare a sample of frame timestamps? ")
    #     if sample.lower() == "yes" or sample.lower() == 'y':
    #         if vid_1.frame_difference_sample==vid_2.frame_difference_sample:
    #             print("Differences match, attempting sync!")
    #         else:
    #             print("Differences do not match, abandoning sync!")
    #             return
    #     else:
    #         return

    if vid_1.fps != vid_2.fps:
        print(
            f"Input videos have different framerates of {vid_1.fps} and {vid_2.fps}. Sync not possible without re-encode.")
        return

    compare_aspect_dim(vid_1, vid_2)

    if vid_2.reference_vid:
        reference_vid = vid_2
        to_sync_vid = vid_1
    else:
        reference_vid = vid_1
        to_sync_vid = vid_2
    workers.threaded_ssim(reference_vid,
                          workers.Worker_Transition, min, frame_window=[0, int(20*reference_vid.fps)])

    quick_match = workers.quick_match_check(
        reference_vid, to_sync_vid)
    if not quick_match:
        workers.threaded_ssim(
            to_sync_vid, workers.Worker_Transition_Match, max, video_source_2=reference_vid, frame_window=[0, int(60*reference_vid.fps)])

    print_find_result(vid_1, vid_2)
    if view:
        match_1 = concat_frames(
            (reference_vid.transition_frames[0], to_sync_vid.transition_frames[0]))
        match_2 = concat_frames(
            (reference_vid.transition_frames[1], to_sync_vid.transition_frames[1]))
        display_frame = concat_frames([match_1, match_2], axis=0)
        view_frame(display_frame)
    if save_frames:

        cv2.imwrite(os.path.join(os.path.dirname(
            sys.argv[0]), "results", reference_vid.name+'.reference_0.jpg'), reference_vid.transition_frames[0])
        cv2.imwrite(os.path.join(os.path.dirname(
            sys.argv[0]), "results", reference_vid.name+'.reference_1.jpg'), reference_vid.transition_frames[1])
        cv2.imwrite(os.path.join(os.path.dirname(
            sys.argv[0]), "results", to_sync_vid.name+'.to_sync_0.jpg'), to_sync_vid.transition_frames[0])
        cv2.imwrite(os.path.join(os.path.dirname(
            sys.argv[0]), "results", to_sync_vid.name+'.to_sync_1.jpg'), to_sync_vid.transition_frames[1])
    return vid_1, vid_2


def batch_find(dir_1, dir_2, view=None, save_frames=None):
    if view is None:
        view = False
    list_1 = sorted((os.path.join(dir_1, f) for f in os.listdir(dir_1)
                    if not f.startswith(".")), key=str.lower)
    list_2 = sorted((os.path.join(dir_2, f) for f in os.listdir(dir_2)
                    if not f.startswith(".")), key=str.lower)

    for i, (file_1, file_2) in enumerate(zip(list_1, list_2)):
        vid_1, vid_2 = find(
            file_1, file_2, view, save_frames)


def main():
    os.makedirs(os.path.join(os.path.dirname(
        sys.argv[0]), "results"), exist_ok=True)
    # batch_find("", "",save_frames=True)
    find(".samples//sample_low_res.mp4",
         "./samples/sample_high_res.mp4", view=True)


if __name__ == "__main__":
    main()
