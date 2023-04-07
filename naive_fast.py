import cv2
from skimage.metrics import structural_similarity as ssim
import threaded_consecutive_ssim_threshold

def naive_compare(file1, file2):
    v1f1, v1f2, v1f1_v1f2_min_ssim, v1f1_number, v1f1_timestamp = threaded_consecutive_ssim_threshold.threaded_ssim(file1,
        threaded_consecutive_ssim_threshold.Worker_Transition,min)
    height, width = v1f1.shape
    dim = width, height

    cap = cv2.VideoCapture(file2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, v1f1_number)
    success, v2_frame = cap.read()

    v2_frame_resized = threaded_consecutive_ssim_threshold.transform_frame(v2_frame,dim=dim)

    threaded_consecutive_ssim_threshold.view_frames(v1f1,v2_frame_resized)

    result = ssim(v1f1, v2_frame_resized)
    print(result)
    if result > .9:
        print('In Sync!')
    else:
        print("Out of Sync!")


def new_double_test(file1, file2):
    v1f1, v1f2, v1f1_v1f2_min_ssim, v1f1_number, v1f1_timestamp = threaded_consecutive_ssim_threshold.threaded_ssim(file1,
        threaded_consecutive_ssim_threshold.Worker_Transition,min)
    result = threaded_consecutive_ssim_threshold.threaded_ssim(file2, threaded_consecutive_ssim_threshold.Worker_Transition_Match, max, v1f1, v1f2)

def new_frame_match_test(file1, file2):
    v1f1, v1f2, v1f1_v1f2_min_ssim, v1f1_number, v1f1_timestamp = threaded_consecutive_ssim_threshold.threaded_ssim(file1,
        threaded_consecutive_ssim_threshold.Worker_Transition,min)
    result = threaded_consecutive_ssim_threshold.threaded_ssim(file2, threaded_consecutive_ssim_threshold.Worker_Frame_Match, max, v1f1)
