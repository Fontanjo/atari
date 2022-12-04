import numpy as np
from brutelogger import BruteLogger
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import os
from pathlib import Path
import dask
import cv2
import time
import random
import itertools
import matplotlib.pyplot as plt

ENV_NAME = 'ALE/Kangaroo-v5'
CONCURRENT_VIDEO = 4
TEMP_VIDEO_FOLDER = "video/temp"
SAVE_PATH = 'dataset_initial/full'
SCHEDULER = "threads" # "processes"
SIMILARITY_THRESHOLD = 800
MINIMAL_ADD = 10
MAX_REPETITIONS = 5
SHUFFLE = True                          # Shuffle frames among each video to avoid favoring initial frames


##############################################################################################################################
# An error occurred when using VideoRecorded, therefore                                                                      #
# I commented the lines 58-64 in                                                                                             #
#  /home/jonas/.local/share/virtualenvs/atari-3zR0bG5k/lib/python3.8/site-packages/gym/wrappers/monitoring/video_recorder.py #
##############################################################################################################################


def get_similarity_quantiles(init, stop=1, step=1):

    # Setup logger
    BruteLogger.save_stdout_to_file(path='logs')

    # Get env name (withouth 'ALE')
    short_env_name = ENV_NAME.split('/')[-1]

    # Store the images to save
    # TODO consider saving them to file as soon as added to this list, instead of doing at the end -> probably not very usefull
    # TODO load from folder if some already saved
    images = load_images(SAVE_PATH + '/' + short_env_name)
    nb_initial_images = len(images)

    # Keep track of added images
    nb_added_images = []

    # Generate new videos
    videos = [record_video(ENV_NAME, record_path=f"{TEMP_VIDEO_FOLDER}/temp_{i}.mp4") for i in range(CONCURRENT_VIDEO)]
    print(f'\n{CONCURRENT_VIDEO} new videos generated')

    # Prepare execution
    lazy_results = []
    for video_path in videos:
        lazy_results.append(dask.delayed(get_new_images)(video_path, shuffle=SHUFFLE, threshold=SIMILARITY_THRESHOLD))

    # Execute
    start = time.time()
    res = dask.compute(lazy_results, scheduler=SCHEDULER)
    # Unzip
    results, len_frames, similarities = zip(*res[0])

    print('Merging similarities')

    similarities_f = list(itertools.chain.from_iterable(similarities))
    # print(len(similarities_f))

    # Get quantiles
    quantiles = np.quantile(similarities_f, np.arange(0.9, 1, 1))
    # print(quantiles)
    #
    # plt.hist([x for x in similarities_f if x < 5000])
    # plt.show()

    # Delete videos
    delete_videos(videos)

    return quantiles



# Play a given environment with a given controller, then record the video
# TODO find a way to return the video without saving it
def record_video(env_name, max_iter=1000, controller='random', record_path="video/temp/temp.mp4"):
    env = gym.make(env_name, render_mode='rgb_array')

    env_name_short = env_name.split('/')[-1]

    env.metadata['render_fps'] = 30

    record = True

    record_dir = '/'.join(record_path.split('/')[:-1])


    if record is not None:
        if record_path is None:
            print('Please specify the path in which saving the video')
            return
        else:
            Path(record_dir).mkdir(parents=True, exist_ok=True)
            video_recorder = VideoRecorder(env, record_path)


    obs = env.reset()

    done = False

    i = 0
    while not done and i < max_iter:
        env.render()
        act = env.action_space.sample()
        obs, rew, done, truncated, info = env.step(act)
        i += 1
        if record: video_recorder.capture_frame()


    env.close()


    if record is not None and record_path is not None:
        video_recorder.close()
        video_recorder.enabled = False

    return record_path



# Delete a list of videos given their path
def delete_videos(video_paths):
    for path in video_paths:
        os.remove(path)
        os.remove(f"{path[:-3]}meta.json")


# Load images from specified path, if any
def load_images(path=SAVE_PATH):
    if not os.path.isdir(path):
        print(f'{path} does not yet exist or is not a dir')
        print(f'Initializing with an empty list')
        return []
    # TODO load images and convert in npy, then divide by 255
    return []



# Take a video path as input and return all the frames contained
def load_frames(video_path, shuffle=True):
    assert video_path.endswith('.mp4'), 'Video format must be mp4!'
    start = time.time()
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
    # Shuffle to avoid favour initial frames
    ### HOWEVER: not shuffling means that comparation will be much faster among frames
    ###  of the same video, so this will considerably slow down the procedure
    random.shuffle(frames)
    stop = time.time()
    print(f'{len(frames)} frames extracted from {video_path} and shuffled in {round(stop - start, 2)}s')
    # Return list of frames
    return frames


# Get the sum of pixelwise difference between two images
def image_difference(img0, img1):
    return np.sum(np.abs(np.subtract(img0, img1)))


# Get the minimal (pixelwise) difference between an image and a list of images
def min_difference_from_list(img_list, img, threshold=1e-2):
    # If the list is empty, return the sum of the image pixels
    if len(img_list) < 1:
        return np.sum(np.abs(img))
    # assert len(img_list) >= 1, 'The img list should have at least 1 image!'
    diffs = []
    for comp_img in img_list:
        new_diff = image_difference(comp_img, img)
        diffs.append(new_diff)
        # If already less than the minimum value, return
        if new_diff < threshold:
            return new_diff
    return min(diffs)


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
    return lst, diff


# Take a video path as input and return all the 'usefull' informations it contains
def get_new_images(video_path, shuffle=True, threshold=1e-2):
    # Load frames from video
    frames = load_frames(video_path, shuffle=shuffle)
    # Initialize list
    lst = []
    # Keep track of similaritis
    similarities = []
    # Add other frames, if they are different enough
    for i in range(len(frames)):
        img = frames[i] / 255
        lst, diff = add_image_to_list(lst, img, threshold)
        similarities.append(diff)
    return lst, len(frames), similarities


# Save images
def save_frames(frames, save_path):
    # Create folder
    os.makedirs(save_path, exist_ok=True)
    i = 0
    for frame in frames:
        out_name = save_path + '/' + str(i) + '.jpg'
        cv2.imwrite(out_name, frame * 255)
        i += 1
    print(f'{i} images saved in {save_path}')




if __name__ == "__main__":
    main(0.9)
