# to do -> handle mismatched aspect ratios and potention cropping, target search based on winning frame from consecutive ssim to priority search in taget ssim
import csv
import os
from fractions import Fraction

import cv2
import matplotlib.pyplot as plt
import numpy
from skimage.metrics import structural_similarity as ssim

from threaded_consecutive_ssim_threshold import \
    threaded_consecutive_ssim_threshold
from threaded_double_target_ssim_threshold import \
    threaded_double_target_ssim_threshold
from threaded_target_ssim_threshold import threaded_target_ssim_threshold

common_fps = 24000/1001


def get_aspect_ratio(file):
    cap = cv2.VideoCapture(file)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    cap.release()
    return width, height, Fraction(width, height)


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


def print_delay(file1, file2, v1f1_v1f2_min_ssim, match_found, v1f1_v2f1_ssim, v1f2_v2f2_ssim, v1f1_number, v2f1_number):

    if not os.path.isfile(os.path.join(os.path.split(file1)[0], 'frame_sync.csv')):
        with open(os.path.join(os.path.split(file1)[0], 'frame_sync.csv'), mode='w') as csvfile:
            csv_writer = csv.writer(
                csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(
                ["File 1", "File 2", "Match Found", "V1F1 Number", "V2F1 Number", "Delay"])

    with open(os.path.join(os.path.split(file1)[0], 'frame_sync.csv'), mode='a') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        print(
            f'The local minimum SSIM between consecutive frames of {os.path.split(file1)[1]} is {v1f1_v1f2_min_ssim}')
        if match_found:
            print(
                f'Match found between {os.path.split(file1)[1]} and {os.path.split(file2)[1]}!')
            print(
                f'The closest match has SSIM {v1f1_v2f1_ssim} between F1s and {v1f2_v2f2_ssim} between F2s')

            frame_difference = v1f1_number - v2f1_number
            if frame_difference > 0:
                print(
                    f'{os.path.split(file1)[1]} is out of sync with {os.path.split(file2)[1]}')
                print(f'V1F1 is number {v1f1_number}')
                print(f'V2F1 is number {v2f1_number}')
                print(
                    f'{os.path.split(file2)[1]} needs to be delayed {frame_difference / common_fps} seconds.\n')
            elif frame_difference < 0:
                print(
                    f'{os.path.split(file1)[1]} is out of sync with {os.path.split(file2)[1]}')
                print(f'V1F1 is number {v1f1_number}')
                print(f'V2F1 is number {v2f1_number}')
                print(
                    f'{os.path.split(file1)[1]} needs to be delayed {-1*frame_difference / common_fps} seconds.\n')
            else:
                print(
                    f'{os.path.split(file1)[1]} is in sync with {os.path.split(file2)[1]}\n')
                print(f'V1F1 is number {v1f1_number}')
                print(f'V2F1 is number {v2f1_number}')

            csv_writer.writerow(
                [file1, file2, match_found, v1f1_number, v2f1_number, frame_difference / common_fps])
        else:
            print(
                f'No frame match found between {os.path.split(file1)[1]} and {os.path.split(file2)[1]}')
            print(
                f'The closest match has SSIM {v1f1_v2f1_ssim} between F1s and {v1f2_v2f2_ssim} between F2s\n')
            csv_writer.writerow([file1, file2, match_found])


def crop_image_only_outside(img, tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img > tol
    m, n = img.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n-mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m-mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


temp = []


# file1 needs to be the higher res if comparing two sources with different resolutions, aspect ratio must match
def find(file1, file2, offset=None):
    if offset == None:
        offset = 0
    # 4:3 aspect ratio - 1440x1080 -> 640x480 is a 2.25 factor reduction in both dimensions

    width1, height1, aspect1 = get_aspect_ratio(file1)
    width2, height2, aspect2 = get_aspect_ratio(file2)

    if aspect1 != aspect2:  # note that threaded_consecutive_ssim_threshold will shrink image dimensions by factor of 10 and convert to int, which will round down, effectively *cropping* the images sahpes one's digit
        print(
            f"Mismatched aspect ratios of {aspect1} and {aspect2}. Cannot use SSIM. Exiting!")
        # print(width1,height1)
        # print(width2,height2)
        # return
    elif width2 > width1:
        print('swap')
        file1, file2 = file2, file1
        width1, height1, aspect1 = get_aspect_ratio(file1)
        width2, height2, aspect2 = get_aspect_ratio(file2)

    scaling_factor = width2 / width1
    # print(scaling_factor)
    dim = int(width1*scaling_factor*.1), int(height1*scaling_factor*.1)
    # print(dim)

    temp_list = []

    v1f1, v1f2, v1f1_v1f2_min_ssim, v1f1_number = threaded_consecutive_ssim_threshold(
        file1)

    # naive quick match check
    quick_match = False
    cap = cv2.VideoCapture(file2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, v1f1_number + offset)
    success, img = cap.read()

    # refactor into it's own function at some point
    height, width, _ = img.shape
    dim = int(width*.1), int(height*.1)
    resize = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    resize = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

    success, img_next = cap.read()
    height_next, width_next, _ = img_next.shape
    dim_next = int(width_next*.1), int(height_next*.1)
    resize_next = cv2.resize(img_next, dim_next, interpolation=cv2.INTER_AREA)
    resize_next = cv2.cvtColor(resize_next, cv2.COLOR_BGR2GRAY)

    result = ssim(v1f1, resize)
    result_next = ssim(v1f2, resize_next)
    if result > .9 and result_next > .9:
        print(f'{file1} is in sync with {file2}, with an offset of {offset} frames!')
        quick_match = True
        temp.append((v1f1, v1f2, resize, resize_next))
    else:
        print(
            f'{file1} is out of sync with {file2}, with an offset of {offset} frames!')
        print(result, result_next)
        temp.append((v1f1, v1f2, resize, resize_next))

    # print(v1f1.shape)
    # print(v1f2.shape)
    # crop1 = crop_image_only_outside(v1f1,tol=0)
    # crop2 = crop_image_only_outside(v1f2,tol=0)
    # crop3 = crop_image_only_outside(v2f1,tol=0)
    # crop4 = crop_image_only_outside(v2f2,tol=0)
    # print(crop1.shape)
    # print(crop2.shape)
    # print(crop3.shape)
    # print(crop4.shape)

    # print(v1f1.shape)
    # print(v1f2.shape)

    if not quick_match:
        # v1f1 = cv2.resize(v1f1, dim, interpolation = cv2.INTER_AREA)
        # v1f2 = cv2.resize(v1f2, dim, interpolation = cv2.INTER_AREA)

        # print(v1f1.shape)
        # print(v1f2.shape)

        v2f1, v2f2, v1f1_v2f1_ssim, v1f2_v2f2_ssim, v2f1_number, match_found = threaded_double_target_ssim_threshold(
            file2, v1f1, v1f2)

        temp_list.extend([file1, file2, v1f1, v1f2, v2f1, v2f2, v1f1_v1f2_min_ssim,
                         v1f1_v2f1_ssim, v1f2_v2f2_ssim, v1f1_number, v2f1_number, match_found])

        match_list.append(temp_list)

        print_delay(file1, file2, v1f1_v1f2_min_ssim, match_found,
                    v1f1_v2f1_ssim, v1f2_v2f2_ssim, v1f1_number, v2f1_number)

        # print(len(match_list))
        # if img is not None:
        #     print(f'Match found! The closest match has SSIM {ssim_value}')
        # else:
        #     print(f'No match found! The closest match has SSIM {ssim_value}')


batch_find = iterate(find)


def find_from_image(file1, file2, image1=None, image2=None, frame_window=None):
    if image1 is None or image2 is None:
        print("Images not provided. Quiting!")
        return
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    height, width, _ = img1.shape
    dim = int(width*.1), int(height*.1)
    img1_resize = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
    img1_resize = cv2.cvtColor(img1_resize, cv2.COLOR_BGR2GRAY)
    img2_resize = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)
    img2_resize = cv2.cvtColor(img2_resize, cv2.COLOR_BGR2GRAY)

    v1f1, v1f2, v1f1_img1_resize_ssim, v1f2_img2_resize_ssim, v1f1_number, match_found1 = threaded_double_target_ssim_threshold(
        file1, img1_resize, img2_resize, frame_window)
    v2f1, v2f2, v2f1_img1_resize_ssim, v2f2_img2_resize_ssim, v2f1_number, match_found2 = threaded_double_target_ssim_threshold(
        file2, img1_resize, img2_resize, frame_window)

    match_found = match_found1 and match_found2
    v1f1_v1f2_min_ssim = ssim(img1_resize, img2_resize)
    v1f1_v2f1_ssim = ssim(v1f1, v2f1)
    v1f2_v2f2_ssim = ssim(v1f2, v2f2)

    temp_list.extend([file1, file2, v1f1, v1f2, v2f1, v2f2, v1f1_v1f2_min_ssim,
                     v1f1_v2f1_ssim, v1f2_v2f2_ssim, v1f1_number, v2f1_number, match_found])
    match_list.append(temp_list)

    print_delay(file1, file2, v1f1_v1f2_min_ssim, match_found,
                v1f1_v2f1_ssim, v1f2_v2f2_ssim, v1f1_number, v2f1_number)


batch_find_from_image = iterate(find_from_image)

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
    file1, file2, v1f1, v1f2, v2f1, v2f2, v1f1_v1f2_min_ssim, v1f1_v2f1_ssim, v1f2_v2f2_ssim, v1f1_number, v2f1_number, match_found = tuple(
        item)
    print_delay(file1, file2, v1f1_v1f2_min_ssim, match_found,
                v1f1_v2f1_ssim, v1f2_v2f2_ssim, v1f1_number, v2f1_number)
    while True:
        cv2.imshow('V1F1', v1f1)
        cv2.imshow('V1F2', v1f2)
        cv2.imshow('V2F1', v2f1)
        cv2.imshow('V2F2', v2f2)
        if cv2.waitKey(1) == ord('q'):
            break


batch_find("", offset=24)
