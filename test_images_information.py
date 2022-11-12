import cv2
import os
import time
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import dask


VIDEO_FOLDER = 'video'
ENV_NAME = "Seaquest-v5"


SIMILARITY_THRESHOLD = 800


def main():
    videos = os.listdir(VIDEO_FOLDER + '/' + ENV_NAME)
    videos = [v for v in videos if v.endswith('.mp4')]

    # Prepare execution
    lazy_results = []
    for video_name in videos:
        lazy_results.append(dask.delayed(get_new_images)(f'{VIDEO_FOLDER}/{ENV_NAME}/{video_name}', SIMILARITY_THRESHOLD))



    # Execute
    start = time.time()
    res = dask.compute(lazy_results, scheduler='processes')
    # Unzip
    results, len_frames = zip(*res[0])
    stop = time.time()

    print(f'Time for extracting frames: {round(stop - start, 2)}s')


    # Merge images from different videos
    start = time.time()
    merged = merge_lists(results, SIMILARITY_THRESHOLD)
    stop = time.time()

    print(f'Time for merging: {round(stop - start, 2)}s')



    # Print some statistics
    print('Frames kept for each video:', [len(r) for r in results])
    print('Frames for each video:', len_frames)
    print('Frames kept at the end: ', len(merged))
    print('Total frames inspected: ', sum(len_frames))

    # Save images
    start = time.time()
    save_frames(merged)
    stop = time.time()

    print(f'Time for saving: {round(stop - start, 2)}s')

# Merge two or more list of images, adding the second/third/fourth/... to the first one. Add only the elements that are not too similar
def merge_lists(list_of_lists, threshold):
    assert len(list_of_lists) > 0
    # Start adding the first one
    # TODO check if can be improved, e.g. adding the shortest (or longest?) one first
    # new_lst = [i / 255 for i in list_of_lists[0]]
    new_lst = list_of_lists[0].copy()
    for lst in list_of_lists[1:]:
        for img in lst:
            new_lst = add_image_to_list(new_lst, img, threshold)
    return new_lst


# Add an image to a list of images if it's not too similar. The image should have values in range [0, 1]
def add_image_to_list(img_list, img, threshold):
    lst = img_list
    diff = min_difference_from_list(lst, img)
    if diff > threshold:
        lst.append(img)
    return lst


# Take a video path as input and return all the 'usefull' informations it contains
def get_new_images(video_path, threshold=1e-2):
    frames = load_frames(video_path)
    # Take first frame
    lst = [frames[0] / 255]
    # Add other frames, if they are different enough
    for i in range(1, len(frames)):
        img = frames[i] / 255
        lst = add_image_to_list(lst, img, threshold)
    return lst, len(frames)



# Take a video path as input and return all the frames contained
def load_frames(video_path):
    assert video_path.endswith('.mp4'), 'Video format must be mp4!'
    # Load video
    video = cv2.VideoCapture(video_path)
    # Store frames
    frames = []
    # Retrive frames
    while(True):
        retrived, frame = video.read()
        if retrived:
            frames.append(frame)
        else:
            break
    # Release all space and windows once done
    video.release()
    cv2.destroyAllWindows()
    # Return list of frames
    return frames


# Get the sum of pixerlwise difference between two images
def image_difference(img0, img1):
    return np.sum(np.abs(np.subtract(img0, img1)))


def min_difference_from_list(img_list, img, threshold=1e-2):
    assert len(img_list) >= 1, 'The img list should have at least 1 image!'
    diffs = []
    for comp_img in img_list:
        new_diff = image_difference(comp_img, img)
        diffs.append(new_diff)
        # If already less than the minimum value, return
        if new_diff < threshold:
            return new_diff
    return min(diffs)



def save_frames(frames):
    image_folder = "dataset_initial"
    # Create folder
    os.makedirs(image_folder + '/' + ENV_NAME, exist_ok=True)
    i = 0
    for frame in frames:
        out_name = image_folder + '/' + ENV_NAME + '/' + str(i) + '.jpg'
        cv2.imwrite(out_name, frame * 255)
        i += 1
    print(f'{i} images saved for env {ENV_NAME}')



if __name__ == "__main__":
    main()
