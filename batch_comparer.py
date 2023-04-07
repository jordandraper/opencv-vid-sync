import os

import cv2
from skimage.metrics import structural_similarity as ssim


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


def compare(original_file, alternate_file):
    # load the images -- the original, the original + alternate,
    # and the original + photoshop
    original = cv2.imread(original_file)
    alternate = cv2.imread(alternate_file)

    # convert the images to grayscale
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    alternate = cv2.cvtColor(alternate, cv2.COLOR_BGR2GRAY)

    print(ssim(original, alternate))


batch_compare = iterate(compare)
