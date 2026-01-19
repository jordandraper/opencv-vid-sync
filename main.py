import csv
import os
import sys
import argparse

import cv2
import numpy as np
import workers
from helpers import (
    concat_frames,
    crop_frame,
    frame_is_letterboxed,
    get_fps_cv_native,
    get_frame_aspect_ratio,
    view_frame,
)

BLACK_THRESHOLD = 90
FPS_FRAME_MULTIPLIER = 20
EXT_FPS_FRAME_MULTIPLIER = 3 * FPS_FRAME_MULTIPLIER


def get_reference_and_sync_videos(first_vid, second_vid):
    if first_vid.reference_vid:
        return first_vid, second_vid
    else:
        return second_vid, first_vid


def write_to_csv(csv_writer, reference_vid, to_sync_vid, timestamp_difference):
    csv_writer.writerow(
        [
            reference_vid.file,
            to_sync_vid.file,
            reference_vid.match_found,
            reference_vid.transition_frames_timestamp,
            to_sync_vid.transition_frames_timestamp,
            timestamp_difference,
        ]
    )


def handle_no_match_found(reference_vid, to_sync_vid):
    print(f"\nNo frame match found between {reference_vid.name} and {to_sync_vid.name}")
    print(
        f"The closest match has SSIM {to_sync_vid.match_frames_ssim} between frames.\n"
    )
    view = input(f"Would you like to manually view? ")
    if view.strip().lower() in ["yes", "y"]:
        view_frame(
            concat_frames(
                (reference_vid.transition_frames[0], to_sync_vid.transition_frames[0])
            )
        )
        manual_inspect = input(f"Are the frames in sync? ")
        if manual_inspect.strip().lower() in ["yes", "y"]:
            reference_vid.match_found = True
            to_sync_vid.match_found = True


def print_find_result(first_vid, second_vid):
    # reference_vid, to_sync_vid = get_reference_and_sync_videos(first_vid, second_vid)
    # I don't think we want this^

    results_dir = os.path.join(os.path.dirname(sys.argv[0]), "results")
    results_file = os.path.join(results_dir, "frame_sync.csv")

    with open(
        results_file, mode="a" if os.path.isfile(results_file) else "w"
    ) as csvfile:
        csv_writer = csv.writer(
            csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        if csvfile.mode == "w":
            csv_writer.writerow(
                [
                    "File 1",
                    "File 2",
                    "Match Found",
                    "V1F1 Timestamp",
                    "V2F1 Timestamp",
                    "Delay",
                ]
            )

        timestamp_difference = (
            reference_vid.transition_frames_timestamp
            - to_sync_vid.transition_frames_timestamp
        )

        if reference_vid.match_found:
            print(f"\nMatch found between {reference_vid.name} and {to_sync_vid.name}!")
            print(
                f"The closest match has SSIM {to_sync_vid.match_frames_ssim} between frames.\n"
            )
            print(
                f"{reference_vid.name} timestamp is {reference_vid.transition_frames_timestamp}ms."
            )
            print(
                f"{to_sync_vid.name} timestamp is {to_sync_vid.transition_frames_timestamp}ms."
            )
            print(
                f"{to_sync_vid.name} needs to be delayed {timestamp_difference} milliseconds.\n"
            )
            write_to_csv(csv_writer, reference_vid, to_sync_vid, timestamp_difference)
        else:
            handle_no_match_found(reference_vid, to_sync_vid)
            write_to_csv(csv_writer, reference_vid, to_sync_vid, timestamp_difference)


class VideoOpenError(Exception):
    """Exception raised when an error occurs opening the video file."""

    pass


class FrameConversionError(Exception):
    """Exception raised when an error occurs converting a frame."""

    pass


class VideoSource:
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
        try:
            cap_1 = cv2.VideoCapture(self.file)
        except cv2.error as e:
            raise VideoOpenError(f"Error opening video file: {e}")
        while True:
            _, v1f = cap_1.read()

            try:
                # converts the frame to gray scale for easier computation
                gray = cv2.cvtColor(v1f, cv2.COLOR_BGR2GRAY)
            except cv2.error as e:
                raise FrameConversionError(f"Error converting frame to grayscale: {e}")

            if np.average(gray) < BLACK_THRESHOLD:
                # skips an iteration, so the frame isn't saved
                continue
            else:
                break
        self.non_black_frame = v1f
        (
            self.letterbox,
            self.crop_height_mask,
            self.crop_width_mask,
        ) = frame_is_letterboxed(v1f)
        # fps can be tricky e.g. 24000/1001 vs 23976/1000 vs 2997/125. Both round to 23.976
        self.fps = round(get_fps_cv_native(self.file), 3)
        # self.frame_difference_sample, self.framerate_type = vfr_cfr_check(self.file)


def set_video_properties(vid_1, vid_2, is_vid_1_scaled):
    vid_1.scale = is_vid_1_scaled
    vid_2.scale = not is_vid_1_scaled
    vid_1.reference_vid = not is_vid_1_scaled
    vid_2.reference_vid = is_vid_1_scaled
    vid_1.to_sync_vid = is_vid_1_scaled
    vid_2.to_sync_vid = not is_vid_1_scaled


def compare_aspect_dim(first_vid, second_vid):
    cropped_v1f = crop_frame(
        first_vid.non_black_frame, first_vid.crop_height_mask, first_vid.crop_width_mask
    )
    cropped_v2f = crop_frame(
        second_vid.non_black_frame,
        second_vid.crop_height_mask,
        second_vid.crop_width_mask,
    )

    width_1, height_1, aspect_1 = get_frame_aspect_ratio(cropped_v1f)
    width_2, height_2, aspect_2 = get_frame_aspect_ratio(cropped_v2f)

    area_1, area_2 = width_1 * height_1, width_2 * height_2

    if area_1 > area_2:
        is_vid_1_scaled = True
        width_scaled, height_scaled = width_2, height_2
    elif area_1 <= area_2:
        is_vid_1_scaled = False
        width_scaled, height_scaled = width_1, height_1
    set_video_properties(first_vid, second_vid, is_vid_1_scaled)

    first_vid.width_scaled = width_scaled
    first_vid.height_scaled = height_scaled
    second_vid.width_scaled = width_scaled
    second_vid.height_scaled = height_scaled


def find(file_1, file_2, view=None, save_frames=None):
    if view is None:
        view = False
    if save_frames is None:
        save_frames = False

    videos = [VideoSource(file_1), VideoSource(file_2)]

    for video in videos:
        try:
            video.set_vid_info()
        except (VideoOpenError, FrameConversionError) as e:
            print(e)
            # Handle error: retry, exit, log, etc.
            # exit the script with a non-zero status to indicate an error
            sys.exit(1)

    if not all(videos[0].fps == video.fps for video in videos):
        print(
            f"Input videos have different framerates. Sync not possible without re-encode."
        )
        return

    compare_aspect_dim(*videos)

    reference_vid, to_sync_vid = get_reference_and_sync_videos(*videos)

    workers.threaded_ssim(
        reference_vid,
        workers.Worker_Transition,
        min,
        frame_window=[0, int(FPS_FRAME_MULTIPLIER * reference_vid.fps)],
    )

    quick_match = workers.quick_match_check(reference_vid, to_sync_vid)
    if not quick_match:
        workers.threaded_ssim(
            to_sync_vid,
            workers.Worker_Transition_Match,
            max,
            video_source_2=reference_vid,
            frame_window=[0, int(EXT_FPS_FRAME_MULTIPLIER * reference_vid.fps)],
        )

    videos = [reference_vid, to_sync_vid]

    print_find_result(*videos)

    if view:
        matches = []
        for i in range(2):
            match = concat_frames(
                (reference_vid.transition_frames[i], to_sync_vid.transition_frames[i])
            )
            matches.append(match)
        display_frame = concat_frames(matches, axis=0)
        view_frame(display_frame)

    if save_frames:
        cwd = os.path.dirname(sys.argv[0])
        for i in range(2):
            cv2.imwrite(
                os.path.join(cwd, "results", f"{reference_vid.name}.reference_{i}.jpg"),
                reference_vid.transition_frames[i],
            )
            cv2.imwrite(
                os.path.join(cwd, "results", f"{to_sync_vid.name}.to_sync_{i}.jpg"),
                to_sync_vid.transition_frames[i],
            )

    return videos


def batch_find(dir_1, dir_2, view=None, save_frames=None):
    if view is None:
        view = False
    list_1 = sorted(
        (os.path.join(dir_1, f) for f in os.listdir(dir_1) if not f.startswith(".")),
        key=str.lower,
    )
    list_2 = sorted(
        (os.path.join(dir_2, f) for f in os.listdir(dir_2) if not f.startswith(".")),
        key=str.lower,
    )

    for i, (file_1, file_2) in enumerate(zip(list_1, list_2)):
        first_vid, second_vid = find(file_1, file_2, view, save_frames)


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Smart Video Sync")

    # Add positional arguments
    parser.add_argument(
        "-s",
        "--source_files",
        action="store",
        help="The source file or directory of files to sync against.",
    )
    parser.add_argument(
        "-t",
        "--target_files",
        action="store",
        help="The target file or directory of files to sync.",
    )

    # Add optional arguments
    parser.add_argument(
        "-b",
        "--batch",
        action="store_true",
        help="Run the sync on a batch of files. Requires passing directories as input.",
    )
    parser.add_argument(
        "-d",
        "--display",
        action="store_true",
        help="Display the matching frames to view.",
    )
    parser.add_argument(
        "-sf",
        "--save-frames",
        action="store_true",
        help="Save the matching frames in the current directory.",
    )

    parser.add_argument(
        "--demo", action="store_true", help="Run the script in demo mode."
    )

    # Parse the arguments
    args = parser.parse_args()

    def run_demo_mode():
        cwd = os.path.dirname(sys.argv[0])
        find(
            os.path.join(cwd, "samples", "sample_low_res.mp4"),
            os.path.join(cwd, "samples", "sample_high_res.mp4"),
            view=True,
        )

    if args.demo:
        print("Running in demo mode")
        run_demo_mode()
    elif args.source_files and args.target_files:
        if args.save_frames:
            cwd = os.path.dirname(sys.argv[0])
            try:
                os.makedirs(os.path.join(cwd, "results"), exist_ok=True)
            except OSError as e:
                print(f"Error creating directory: {e}")
                print(
                    "Exiting. Try turning off the --save-frames flag or checking that the results directory doesn't already exist."
                )
                sys.exit(
                    1
                )  # exit the script with a non-zero status to indicate an error
        if args.batch:
            batch_find(
                args.source_files, args.target_files, args.display, args.save_frames
            )
        else:
            find(args.source_files, args.target_files, args.display, args.save_frames)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
