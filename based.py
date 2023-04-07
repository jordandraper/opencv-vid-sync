import csv
import os
import subprocess
import tempfile
import time

import cv2
import ffmpeg
import imutils
import numpy
from imutils.video import FileVideoStream
from skimage.metrics import structural_similarity as ssim

# import decord as de
from timer import Timer


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


def sorting_key(string):
    return (int(string.split('.')[1]))

# get video fps as fraction and return it as a tuple of ints


def get_fps(file):
    ffprobe_json = ffmpeg.probe(file, v='error', select_streams='v')
    fps = ffprobe_json['streams'][0]['r_frame_rate']
    fps_num, fps_den = fps.split('/')
    return int(fps_num), int(fps_den)

# frame extraction with frame number as suffix of filename


def extract_frames(file, output_dir=None, start_frame=None, end_frame=None, start_time=None, end_time=None, keyframe=None):

    fps_num, fps_den = get_fps(file)

    # pass start,end time as seconds and if using end_frame be sure to know that it will extract to n-1 not n
    if output_dir is None:
        output_dir = os.getcwd()
    if start_frame is None:
        start_frame = 0
    if start_time is None:
        start_time = start_frame / (fps_num / fps_den)
    if end_time is None:
        if end_frame is None:
            end_time = float(ffmpeg.probe(file, v='error', select_streams='v',
                             show_entries='format=duration')['format']['duration'])
        else:
            # using end_frame instead of end_frame+1 ensures parallel structure to slice mechanic
            end_time = (end_frame) / (fps_num / fps_den)
    if keyframe is None:
        keyframe = True

    if keyframe:
        # filtering through only keyframes makes the extraction much faster
        input = ffmpeg.input(file, skip_frame='nokey',
                             ss=start_time, to=end_time)
        # no need to extract a portion of frames if using keyframes
        out = ffmpeg.output(input, os.path.join(
            output_dir, 'out.%d.jpg'), vsync='0', qscale='1', frame_pts='true')
    else:
        input = ffmpeg.input(file, ss=start_time,
                             to=end_time)  # gets all frames
        # extracts exactly number_of_frames by calculating stopping time
        out = ffmpeg.output(input, os.path.join(
            output_dir, 'out.%d.jpg'), vsync='0', qscale='1', frame_pts='true')

    ffmpeg.run(out, quiet=True)  # quiet supresses stdout,stderr printing


def frame_sync(file1, file2, number_of_frames=None, mode=None):

    if mode is None:
        mode = 'default'

    file1_fps_num, file1_fps_den = get_fps(file1)
    file2_fps_num, file2_fps_den = get_fps(file2)

    file1_fps = file1_fps_num / file1_fps_den
    file2_fps = file2_fps_num / file2_fps_den

    match_found = False

    if file1_fps != file2_fps:
        print('Warning: video FPS do not match. Exiting.')
        return
    else:
        common_fps = file1_fps

    if mode == 'default':
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            extract_frames(file1, tmpdir1, keyframe=False)
            extract_frames(file2, tmpdir2, keyframe=False)

            onlyfiles1 = sorted([f for f in os.listdir(tmpdir1) if os.path.isfile(
                os.path.join(tmpdir1, f)) and not f.startswith('.')], key=sorting_key)
            onlyfiles2 = sorted([f for f in os.listdir(tmpdir2) if os.path.isfile(
                os.path.join(tmpdir2, f)) and not f.startswith('.')], key=sorting_key)

            for f1 in onlyfiles1:
                # load the image and convert to grayscale
                original = cv2.imread(os.path.join(tmpdir1, f1))
                original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                # if cv2.countNonZero(original) == 0:
                #     break
                # print(f'{f1} is completely black.')

                for f2 in onlyfiles2:
                    # load the image and convert to grayscale
                    alternate = cv2.imread(os.path.join(tmpdir2, f2))
                    alternate = cv2.cvtColor(alternate, cv2.COLOR_BGR2GRAY)
                    # if cv2.countNonZero(alternate) == 0:
                    #     break
                    s = ssim(original, alternate)
                    if s > .90:  # in practice .90 is probably a good lower bound, if it's too high then a match might not be found, too low and you get a false positive
                        frame_difference = int(
                            f1.split('.')[1]) - int(f2.split('.')[1])
                        if frame_difference > 0:
                            print(
                                f'{os.path.split(file1)[1]} is out of sync with {os.path.split(file2)[1]}')
                            print(
                                f'{os.path.split(file2)[1]} needs to be delayed {frame_difference / common_fps} seconds.')
                            print(
                                f'Frame match ssim score is {s*100}% between {f1} and {f2}\n')
                        elif frame_difference < 0:
                            print(
                                f'{os.path.split(file1)[1]} is out of sync with {os.path.split(file2)[1]}')
                            print(
                                f'{os.path.split(file1)[1]} needs to be delayed {-1*frame_difference / common_fps} seconds.')
                            print(
                                f'Frame match ssim score is {s*100}% between {f1} and {f2}\n')
                        else:
                            print(
                                f'{os.path.split(file1)[1]} is in sync with {os.path.split(file2)[1]}')
                            print(
                                f'Frame match ssim score is {s*100}% between {f1} and {f2}\n')
                        break  # exits onlyfles2 search after first high probability match
                    else:
                        print(
                            f'No frame match found between {os.path.split(file1)[1]} and {os.path.split(file2)[1]}\n')
                break  # exits onlyfles1 search after first high probability match

    if mode == 'assisted':
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:

            start_frame = quick_binned_threshold_ssim(file1)

            # start_frame+2 ensures that 2 frames are extracted: start_frame and start_frame+1
            extract_frames(file1, tmpdir1, start_frame=start_frame,
                           end_frame=start_frame+2, keyframe=False)
            extract_frames(file2, tmpdir2, keyframe=False)

            onlyfiles1 = sorted([f for f in os.listdir(tmpdir1) if os.path.isfile(
                os.path.join(tmpdir1, f)) and not f.startswith('.')], key=sorting_key)
            onlyfiles2 = sorted([f for f in os.listdir(tmpdir2) if os.path.isfile(
                os.path.join(tmpdir2, f)) and not f.startswith('.')], key=sorting_key)

            frame1_filename = onlyfiles1[0]
            frame2_filename = onlyfiles1[1]

            # load the image and convert to grayscale
            file1_frame1 = cv2.imread(os.path.join(tmpdir1, frame1_filename))
            file1_frame1 = cv2.cvtColor(file1_frame1, cv2.COLOR_BGR2GRAY)

            file1_frame2 = cv2.imread(os.path.join(tmpdir1, frame2_filename))
            file1_frame2 = cv2.cvtColor(file1_frame2, cv2.COLOR_BGR2GRAY)

            # if cv2.countNonZero(original) == 0:
            #     break
            # print(f'{f1} is completely black.')

            for index, frame1_comp_filename in enumerate(onlyfiles2):
                # load the image and convert to grayscale
                file2_frame1 = cv2.imread(
                    os.path.join(tmpdir2, frame1_comp_filename))
                file2_frame1 = cv2.cvtColor(file2_frame1, cv2.COLOR_BGR2GRAY)

                try:
                    frame2_comp_filename = onlyfiles2[index+1]
                    file2_frame2 = cv2.imread(
                        os.path.join(tmpdir2, frame2_comp_filename))
                    file2_frame2 = cv2.cvtColor(
                        file2_frame2, cv2.COLOR_BGR2GRAY)
                except:
                    break
                # if cv2.countNonZero(alternate) == 0:
                #     break
                s1 = ssim(file1_frame1, file2_frame1)
                s2 = ssim(file1_frame2, file2_frame2)
                if s1 > .90 and s2 > .90:  # in practice .90 is probably a good lower bound, if it's too high then a match might not be found, too low and you get a false positive
                    match_found = True
                    frame_difference = start_frame - \
                        int(frame1_comp_filename.split('.')[1])
                    if frame_difference > 0:
                        print(
                            f'{os.path.split(file1)[1]} is out of sync with {os.path.split(file2)[1]}')
                        print(
                            f'{os.path.split(file2)[1]} needs to be delayed {frame_difference / common_fps} seconds.')
                        print(
                            f'Frame match ssim score is {s1*100}% between {frame1_filename} and {frame1_comp_filename}\n')
                        print(
                            f'Frame match ssim score is {s2*100}% between {frame2_filename} and {frame2_comp_filename}\n')
                    elif frame_difference < 0:
                        print(
                            f'{os.path.split(file1)[1]} is out of sync with {os.path.split(file2)[1]}')
                        print(
                            f'{os.path.split(file1)[1]} needs to be delayed {-1*frame_difference / common_fps} seconds.')
                        print(
                            f'Frame match ssim score is {s1*100}% between {frame1_filename} and {frame1_comp_filename}\n')
                        print(
                            f'Frame match ssim score is {s2*100}% between {frame2_filename} and {frame2_comp_filename}\n')
                    else:
                        print(
                            f'{os.path.split(file1)[1]} is in sync with {os.path.split(file2)[1]}')
                        print(
                            f'Frame match ssim score is {s1*100}% between {frame1_filename} and {frame1_comp_filename}\n')
                        print(
                            f'Frame match ssim score is {s2*100}% between {frame2_filename} and {frame2_comp_filename}\n')
                    break  # exits onlyfles2 search after first high probability match
            if not match_found:
                print(
                    f'No frame match found between {os.path.split(file1)[1]} and {os.path.split(file2)[1]}\n')


def consecutive_frame_ssim_compare(path):
    ssim_compare_list = []
    onlyfiles = sorted([f for f in os.listdir(path) if os.path.isfile(
        os.path.join(path, f)) and not f.startswith('.')], key=sorting_key)

    with open(os.path.join(path, 'ssim_compare.csv'), mode='w') as ssim_compare:
        ssim_writer = csv.writer(
            ssim_compare, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        ssim_writer.writerow(['Frame Name', 'Frame Name', 'SSIM'])

        for index, file in enumerate(onlyfiles):
            file1 = file
            try:
                file2 = onlyfiles[index+1]
                s = frame_ssim_compare(os.path.join(
                    path, file1), os.path.join(path, file2))
                ssim_writer.writerow([file1, file2, s])
                ssim_compare_list.append([file1, file2, s])
            except IndexError:
                print('Finished comparison!')

    minimum_ssim = ssim_compare_list[0]
    for item in ssim_compare_list:
        if item[2] < minimum_ssim[2]:
            minimum_ssim = item

    print(minimum_ssim)
    return minimum_ssim


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


frame_sync_batch = iterate(frame_sync)


def quick_binned_threshold_ssim(file1):
    frame_number_shift = 0
    keyframe_list = ffmpeg.probe(file1, loglevel='error', skip_frame='nokey',
                                 select_streams='v:0', show_entries='frame=pkt_pts_time')['frames']
    for index, entry in enumerate(keyframe_list):
        with tempfile.TemporaryDirectory() as tmpdir1:
            try:
                extract_frames(file1, tmpdir1, start_time=float(entry['pkt_pts_time']), end_time=float(
                    keyframe_list[index+1]['pkt_pts_time']), keyframe=False)
                onlyfiles1 = sorted([f for f in os.listdir(tmpdir1) if os.path.isfile(
                    os.path.join(tmpdir1, f)) and not f.startswith('.')], key=sorting_key)
                bin_minimum = consecutive_frame_ssim_compare(tmpdir1)
                local_minimum_frame_number = int(
                    bin_minimum[0].split('.')[1])+frame_number_shift
                if bin_minimum[2] < .6:
                    print(local_minimum_frame_number)
                    return local_minimum_frame_number
                frame_number_shift += int(onlyfiles1[-1].split('.')[1]) + 1
            except IndexError:
                extract_frames(file1, tmpdir1, start_time=float(
                    entry['pkt_pts_time']), keyframe=False)
                onlyfiles1 = sorted([f for f in os.listdir(tmpdir1) if os.path.isfile(
                    os.path.join(tmpdir1, f)) and not f.startswith('.')], key=sorting_key)
                bin_minimum = consecutive_frame_ssim_compare(tmpdir1)
                local_minimum_frame_number = int(
                    bin_minimum[0].split('.')[1])+frame_number_shift
                if bin_minimum[2] < .6:
                    print(local_minimum_frame_number)
                    return local_minimum_frame_number
                frame_number_shift += int(onlyfiles1[-1].split('.')[1]) + 1

                # subprocess.Popen(["open", tmpdir1])
                # input("Press Enter to continue...")


def test(file):
    t = Timer()
    t.start()
    min_ssim = 1
    cap = cv2.VideoCapture(file)
    while cap.isOpened():
        success, img = cap.read()
        success, img_next = cap.read()
        if not success:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        height, width, layers = img.shape
        height_next, width_next, layers_next = img_next.shape
        img_new_w, img_next_new_w = int(width*.1), int(width_next*.1)
        img_new_h, img_next_new_h = int(height*.1), int(height_next*.1)
        dim = img_new_w, img_new_h
        dim_next = img_next_new_w, img_next_new_h
        resize = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        resize_next = cv2.resize(
            img_next, dim_next, interpolation=cv2.INTER_AREA)
        current_ssim = frame_ssim_compare(resize, resize_next)
        if current_ssim < min_ssim:
            min_ssim = current_ssim
            print(min_ssim)
            cv2.imshow('first', resize)
            cv2.imshow('second', resize_next)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    t.stop()
    return resize, resize_next


def super_test(file1, file2):
    t = Timer()
    t.start()
    frame1, frame2 = test(file1)
    frame1_compare, frame2_compare = test(file2)
    s1 = frame_ssim_compare(frame1, frame1_compare)
    s2 = frame_ssim_compare(frame2, frame2_compare)
    print(s1, s2)
    t.stop()


def test_thread(file):
    t = Timer()
    t.start()
    min_ssim = 1
    # start the file video stream thread and allow the buffer to
    # start to fill
    print("[INFO] starting video file thread...")
    fvs = FileVideoStream(file, queue_size=256).start()
    # loop over frames from the video file stream
    while fvs.more():
        frame = fvs.read()
        frame_next = fvs.read()  # need to fix this, i think it is just going 2 at a time i.e. missing comparison between 1 and 2 since it goes 0,1 then 2,3
        height, width, layers = frame.shape
        height_next, width_next, layers_next = frame_next.shape
        img_new_w, img_next_new_w = int(width*.1), int(width_next*.1)
        img_new_h, img_next_new_h = int(height*.1), int(height_next*.1)
        dim = img_new_w, img_new_h
        dim_next = img_next_new_w, img_next_new_h
        resize = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        resize_next = cv2.resize(frame_next, dim_next,
                                 interpolation=cv2.INTER_AREA)
        current_ssim = frame_ssim_compare(resize, resize_next)
        if current_ssim < min_ssim:
            min_ssim = current_ssim
            print(min_ssim)
            cv2.imshow('first', resize)
            cv2.imshow('second', resize_next)
        if cv2.waitKey(1) == ord('q'):
            break
    fvs.stop()
    cv2.destroyAllWindows()
    t.stop()
    return resize, resize_next
