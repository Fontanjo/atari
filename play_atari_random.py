import gym
from pathlib import Path
import numpy as np
import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from utils import get_full_atari_list, get_selected_atari_list

##############################################################################################################################
# An error occurred when using VideoRecorded, therefore                                                                      #
# I commented the lines 58-64 in                                                                                             #
#  /home/jonas/.local/share/virtualenvs/atari-3zR0bG5k/lib/python3.8/site-packages/gym/wrappers/monitoring/video_recorder.py #
##############################################################################################################################

nb_plays = 20

def main():
    for _ in range(nb_plays):
        # env_name = get_random_env(selected_only=True)
        env_name = 'ALE/Seaquest-v5'    # Force specific env
        play_env(env_name)


def play_env(env_name):
    max_iter = 1000

    print(f'Playing {env_name}')

    env = gym.make(env_name, render_mode='rgb_array')

    env_name_short = env_name.split('/')[-1]
    record = True
    record_path = f'video/{env_name_short}'
    os.makedirs(record_path, exist_ok=True)

    existing_files = int(len(os.listdir(record_path)) / 2)

    if record is not None:
        if record_path is None:
            print('Please specify the path in which saving the video')
            return
        else:
            Path(record_path).mkdir(parents=True, exist_ok=True)
            video_recorder = VideoRecorder(env, f"{record_path}/{existing_files}_{env_name_short}.mp4")


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
        print("Video saved")
        video_recorder.close()
        video_recorder.enabled = False

    print('End\n\n')




def get_random_env(selected_only=False):
    if selected_only:
        lst = get_selected_atari_list()
    else:
        lst = get_full_atari_list()

    # Return random env
    return lst[np.random.randint(len(lst))]



if __name__ == "__main__":
    main()
