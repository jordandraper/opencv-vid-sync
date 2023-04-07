# to do -> handle mismatched aspect ratios and potention cropping, target search based on winning frame from consecutive ssim to priority search in taget ssim
# start brute force search within a window of original timestamp, then expand on each failure and avoid previously searched region
import csv
import os
from fractions import Fraction
import numpy as np
import cv2
import ffmpeg
import threaded_consecutive_ssim_threshold
from imutils.video import FileVideoStream
from skimage.metrics import structural_similarity as ssim
# from threaded_double_target_ssim_threshold import \
#     threaded_double_target_ssim_threshold
# from threaded_target_ssim_threshold import threaded_target_ssim_threshold

# from threaded_consecutive_ssim_threshold import \
#     threaded_consecutive_ssim_threshold
from timer import Timer
from pprint import pprint
common_fps = 24000/1001

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

# get video fps as fraction and return it as a tuple of ints

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


match_list = []


def print_delay(file_1, file_2, v1f1_v1f2_min_ssim, match_found, v1f1_v2f1_ssim, v1f1_number, v2f1_number):

    if not os.path.isfile(os.path.join(os.path.split(file_1)[0], 'frame_sync.csv')):
        with open(os.path.join(os.path.split(file_1)[0], 'frame_sync.csv'), mode='w') as csvfile:
            csv_writer = csv.writer(
                csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(
                ["File 1", "File 2", "Match Found", "V1F1 Number", "V2F1 Number", "Delay"])

    with open(os.path.join(os.path.split(file_1)[0], 'frame_sync.csv'), mode='a') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        print(
            f'The local minimum SSIM between consecutive frames of {os.path.split(file_1)[1]} is {v1f1_v1f2_min_ssim}')
        if match_found:
            print(
                f'Match found between {os.path.split(file_1)[1]} and {os.path.split(file_2)[1]}!')
            print(
                f'The closest match has SSIM {v1f1_v2f1_ssim} between F1s.')

            frame_difference = v1f1_number - v2f1_number
            if frame_difference > 0:
                print(
                    f'{os.path.split(file_1)[1]} is out of sync with {os.path.split(file_2)[1]}')
                print(f'V1F1 is number {v1f1_number}')
                print(f'V2F1 is number {v2f1_number}')
                print(
                    f'{os.path.split(file_2)[1]} needs to be delayed {frame_difference / common_fps} seconds.\n')
            elif frame_difference < 0:
                print(
                    f'{os.path.split(file_1)[1]} is out of sync with {os.path.split(file_2)[1]}')
                print(f'V1F1 is number {v1f1_number}')
                print(f'V2F1 is number {v2f1_number}')
                print(
                    f'{os.path.split(file_1)[1]} needs to be delayed {-1*frame_difference / common_fps} seconds.\n')
            else:
                print(
                    f'{os.path.split(file_1)[1]} is in sync with {os.path.split(file_2)[1]}\n')
                print(f'V1F1 is number {v1f1_number}')
                print(f'V2F1 is number {v2f1_number}')

            csv_writer.writerow(
                [file_1, file_2, match_found, v1f1_number, v2f1_number, frame_difference / common_fps])
        else:
            print(
                f'No frame match found between {os.path.split(file_1)[1]} and {os.path.split(file_2)[1]}')
            print(
                f'The closest match has SSIM {v1f1_v2f1_ssim} between F1s.\n')
            csv_writer.writerow([file_1, file_2, match_found])

temp = []

def get_frame_aspect_ratio(frame):

    height, width, _ = frame.shape
    # if cap.isOpened():
    #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    # cap.release()
    return width, height, Fraction(width, height)

# https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
# def frame_is_letterboxed(img):
#     # Read the image, convert it into grayscale, and make in binary image for threshold value of 1.
#     # img = cv2.imread('sofwin.png')
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     _,thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)

#     # Now find contours in it. There will be only one object, so find bounding rectangle for it.
#     contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         cnt = contours[0]
#         x,y,w,h = cv2.boundingRect(cnt)
#         print(x,y,w,h)
        
#         # Now crop the image, and save it into another file.
#         crop_height_mask = slice(y,y+h)
#         crop_width_mask = slice(x,x+w)
#         # cv2.imwrite('sofwinres.png',crop)
#         return True, crop_height_mask,crop_width_mask
#     else:
#         return False, slice(None),slice(None)

def frame_is_letterboxed(img,th=25):
    # gray = cv2.cvtColor(im
    # gray,1,255,cv2.THRESH_BINARY)
    y_nonzero, x_nonzero,_ = np.nonzero(img>th)
    crop_height_mask = slice(np.min(y_nonzero),np.max(y_nonzero)+1)
    crop_width_mask = slice(np.min(x_nonzero),np.max(x_nonzero)+1)
    print(crop_height_mask)
    print(crop_width_mask)
    print("np averages")
    print(np.average(img[crop_height_mask,crop_width_mask])/np.average(img))
    input()
    if np.array_equal(img[crop_height_mask,crop_width_mask],img):
        letterboxed=False
        return letterboxed, slice(None),slice(None)
    else:
        letterboxed=True
        return letterboxed, crop_height_mask, crop_width_mask

    # Now find contours in it. There will be only one object, so find bounding rectangle for it.
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        print(x,y,w,h)
        
        # Now crop the image, and save it into another file.
        crop_height_mask = slice(y,y+h)
        crop_width_mask = slice(x,x+w)
        # cv2.imwrite('sofwinres.png',crop)
        return True, crop_height_mask,crop_width_mask
    else:
        return False, slice(None),slice(None)

def crop_frame(img,crop_height_mask,crop_width_mask):
    return img[crop_height_mask,crop_width_mask]


def get_dim_scaler(file_1,file_2):
    pass
    # if aspect1 != aspect2:  # note that threaded_consecutive_ssim_threshold will shrink image dimensions by factor of 10 and convert to int, which will round down, effectively *cropping* the images sahpes one's digit
    #     print(
    #         f"Mismatched aspect ratios of {aspect1} and {aspect2}. Cannot use SSIM. Exiting!")
    #     return

class VideoSource():
    def __init__(self,file) -> None:
        self.file = file
        self.letterbox = None
        self.crop_height_mask = slice(None)
        self.crop_width_mask = slice(None)
        self.width_scaled = None
        self.height_scaled = None
        self.scale = None
        self.frames = []
        self.non_black_frame = None
    
    def set_vid_info(self):
        """Get first non-black frame"""
        cap_1 = cv2.VideoCapture(self.file)
        while True:
            _, v1f = cap_1.read()
            # converts the frame to gray scale for easier computation
            gray = cv2.cvtColor(v1f, cv2.COLOR_BGR2GRAY)
            # print(np.average(gray))

            if np.average(gray) < 90:
                # skips an iteration, so the frame isn't saved
                continue
            else:
                break
        self.non_black_frame = v1f
        self.letterbox,self.crop_height_mask,self.crop_width_mask = frame_is_letterboxed(v1f)
        

def compare_asprat_dim(vid_1,vid_2):
    cropped_v1f = crop_frame(vid_1.non_black_frame,vid_1.crop_height_mask,vid_1.crop_width_mask)
    cropped_v2f = crop_frame(vid_2.non_black_frame,vid_2.crop_height_mask,vid_2.crop_width_mask)

    width1, height1, aspect1 = get_frame_aspect_ratio(cropped_v1f)
    print(width1, height1)
    width2, height2, aspect2 = get_frame_aspect_ratio(cropped_v2f)
    print(width2, height2)
    input()
    if width1*height1 > width2*height2:
        width_scaled = width2
        height_scaled = height2
        vid_1.scale = True
        vid_2.scale = False
    elif width1*height1 < width2*height2:
        width_scaled = width1
        height_scaled = height1
        vid_1.scale = False
        vid_2.scale = True
    else:
        width_scaled = width1
        height_scaled = height1
        vid_1.scale = False
        vid_2.scale = False
    
    vid_1.width_scaled = width_scaled
    vid_1.height_scaled = height_scaled
    vid_2.width_scaled = width_scaled
    vid_2.height_scaled = height_scaled

def view_frame(frame):
    while True:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

# file_1 needs to be the higher res if comparing two sources with different resolutions, aspect ratio must match
def find(file_1, file_2, offset=None):
    if offset is None:
        offset = 0
    # 4:3 aspect ratio - 1440x1080 -> 640x480 is a 2.25 factor reduction in both dimensions
    temp_list = []

    vid_1 = VideoSource(file_1)
    vid_2 = VideoSource(file_2)

    vid_1.set_vid_info()
    vid_2.set_vid_info()
    compare_asprat_dim(vid_1,vid_2)
    pprint(vars(vid_1))
    # view_frame(crop_frame(vid_1.non_black_frame,vid_1.crop_height_mask,vid_1.crop_width_mask))

    pprint(vars(vid_2))
    # view_frame(crop_frame(vid_2.non_black_frame,vid_2.crop_height_mask,vid_2.crop_width_mask))
    # input()

    #take frame from smaller res video, find frame in higher res video
    # width_scaled, height_scaled, v1f_letterbox, v2f_letterbox = check_asp_dim(file_1, file_2)
    # print(width_scaled, height_scaled, video_to_scale, v1f_letterbox, v2f_letterbox)
    # input()

    # vid_1.letterbox = v1f_letterbox
    # vid_1.width_scaled = width_scaled
    # vid_1.height_scaled = height_scaled

    # vid_2.letterbox = v2f_letterbox
    # vid_2.width_scaled = width_scaled
    # vid_2.height_scaled = height_scaled

    # if video_to_scale == vid_1.file:
    #     vid_1.scale = True
    #     vid_2.scale = False
    # elif video_to_scale == vid_2.file:
    #     vid_1.scale = False
    #     vid_2.scale = True
    # else:
    #     vid_1.scale = False
    #     vid_2.scale = False

    if vid_1.scale:
        v1f1, v1f2, v1f1_v1f2_min_ssim, v1f1_number, v1f1_timestamp = threaded_consecutive_ssim_threshold.threaded_ssim(vid_2,
        threaded_consecutive_ssim_threshold.Worker_Transition,min)
        vid_2.frames.extend([v1f1, v1f2])
        high_res_vid = vid_1
        low_res_vid = vid_2
    else:
        v1f1, v1f2, v1f1_v1f2_min_ssim, v1f1_number, v1f1_timestamp = threaded_consecutive_ssim_threshold.threaded_ssim(vid_1,
        threaded_consecutive_ssim_threshold.Worker_Transition,min)
        vid_1.frames.extend([v1f1, v1f2])
        high_res_vid = vid_2
        low_res_vid = vid_1
    # else:
    #     high_res_vid = None
    #     v1f1, v1f2, v1f1_v1f2_min_ssim, v1f1_number, v1f1_timestamp = threaded_consecutive_ssim_threshold.threaded_ssim(file_1,
    #         threaded_consecutive_ssim_threshold.Worker_Transition,min)

    # naive quick match check
    quick_match = False
    # cap = cv2.VideoCapture(file_2)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, v1f1_number + offset)
    # v2f1_success, v2f1 = cap.read()
    # v2f2_success, v2f2 = cap.read()
    # if v2f1_success and v2f2_success:

    #     if v1f_letterbox:
    #         _, v1f1 = crop_frame_letterbox(v1f1)
    #         _, v1f2 = crop_frame_letterbox(v1f2)
    #     if v2f_letterbox:
    #         _, v2f1 = crop_frame_letterbox(v2f1)
    #         _, v2f2 = crop_frame_letterbox(v2f2)

    #     if video_to_scale == 0:
    #         v1f1 = threaded_consecutive_ssim_threshold.resize_frame(v1f1,dim=(width_scaled,height_scaled))
    #         v1f2 = threaded_consecutive_ssim_threshold.resize_frame(v1f2,dim=(width_scaled,height_scaled))
    #     elif video_to_scale == 1:
    #         v2f1 = threaded_consecutive_ssim_threshold.resize_frame(v2f1,dim=(width_scaled,height_scaled))
    #         v2f2 = threaded_consecutive_ssim_threshold.resize_frame(v2f2,dim=(width_scaled,height_scaled))
    #     print(width_scaled,height_scaled)
    #     print(v1f1.shape)
    #     print(v2f1.shape)
    #     result = threaded_consecutive_ssim_threshold.check_frame_match(v1f1, v2f1)
    #     result_next = threaded_consecutive_ssim_threshold.check_frame_match(v1f2, v2f2)

    #     if result and result_next:
    #         print(f'{file_1} is in sync with {file_2}, with an offset of {offset} frames!')
    #         quick_match = True
    #         temp.append((v1f1, v1f2))
    #     else:
    #         print(
    #             f'{file_1} is out of sync with {file_2}, with an offset of {offset} frames!')
    #         print(result, result_next)
    #         temp.append((v1f1, v1f2))

    if not quick_match:
        v2f1, v2f2, v1f1_v2f1_ssim, v2f1_number, v2f1_timestamp, match_found = threaded_consecutive_ssim_threshold.threaded_ssim(high_res_vid, threaded_consecutive_ssim_threshold.Worker_Transition_Match, max, video_source_2=low_res_vid)
        temp_list.extend([file_1, file_2, v1f1, v1f2, v2f1, v2f2, v1f1_v1f2_min_ssim,
                         v1f1_v2f1_ssim, v1f1_number, v2f1_number, match_found])

        match_list.append(temp_list)

        print_delay(file_1, file_2, v1f1_v1f2_min_ssim, match_found,
                    v1f1_v2f1_ssim, v1f1_number, v2f1_number)

        # print(len(match_list))
        # if img is not None:
        #     print(f'Match found! The closest match has SSIM {ssim_value}')
        # else:
        #     print(f'No match found! The closest match has SSIM {ssim_value}')


batch_find = iterate(find)

for item in temp:
    v1f1, v1f2, v2f1, v2f2 = item
    while True:
        cv2.imshow('V1F1', v1f1)
        cv2.imshow('V1F2', v1f2)
        cv2.imshow('V2F1', v2f1)
        cv2.imshow('V2F2', v2f2)
        if cv2.waitKey(1) == ord('q'):
            break

for item in match_list:
    file_1, file_2, v1f1, v1f2, v2f1, v2f2, v1f1_v1f2_min_ssim, v1f1_v2f1_ssim, v1f2_v2f2_ssim, v1f1_number, v2f1_number, match_found = tuple(
        item)
    print_delay(file_1, file_2, v1f1_v1f2_min_ssim, match_found,
                v1f1_v2f1_ssim, v1f2_v2f2_ssim, v1f1_number, v2f1_number)
    while True:
        cv2.imshow('V1F1', v1f1)
        cv2.imshow('V1F2', v1f2)
        cv2.imshow('V2F1', v2f1)
        cv2.imshow('V2F2', v2f2)
        if cv2.waitKey(1) == ord('q'):
            break