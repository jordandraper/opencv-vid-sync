import cv2
from skimage.metrics import structural_similarity as ssim

from threaded_consecutive_ssim_threshold import \
    threaded_consecutive_ssim_threshold


def naive_compare(file1, file2):
    v1f1, v1f2, v1f1_v1f2_min_ssim, v1f1_number, v1f1_timestamp = threaded_consecutive_ssim_threshold(
        file1)
    cap = cv2.VideoCapture(file2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, v1f1_number)
    success, img = cap.read()
    height, width = v1f1.shape

    dim = int(width), int(height)
    resize = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    resize = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    print(v1f1.shape)
    print(resize.shape)
    while cap.isOpened():
        cv2.imshow('v1f1',v1f1)
        cv2.imshow('resize',resize)
        if cv2.waitKey(1) == ord('q'):
                break
    result = ssim(v1f1, resize)
    print(result)
    if result > .9:
        print('In Sync!')
    else:
        print("Out of Sync!")


naive_compare(,)
