import gym
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import get_full_atari_list, get_selected_atari_list

def main():

    out_path = './summary_selected.csv'

    cols = ['EnvName', 'inputs_space', 'nactions', 'action_space', 'discrete']
    df = pd.DataFrame(columns=cols)

    for env_name in tqdm(get_selected_atari_list()):
        # Get name
        name_short = env_name.split('/')[-1]
        # Create env
        env = gym.make(env_name)
        # Check if discrete
        discrete = type(env.action_space) == gym.spaces.discrete.Discrete
        # Gen nb. of actions
        if discrete:
            nactions = env.action_space.n
        else:
            nactions = env.action_space.shape[0]

        # Add to df
        df = pd.concat([df, pd.DataFrame(data=[[name_short, env.observation_space, nactions, env.action_space, discrete]], columns=cols)])
        # df.append([name_short, env.observation_space, nactions, env.action_space, discrete])

    print(df)
    df.to_csv(out_path)


#
# def get_atari_list():
#     # Get all Arcade Learning Environment (ALE) envs
#     lst = [x for x in list(gym.envs.registry) if 'ALE' in x]
#     # Remove RAM versions
#     lst = [x for x in lst if 'ram' not in x]
#
#     # Remove env not working
#     lst = [x for x in lst if 'TicTacToe3D' not in x]
#     lst = [x for x in lst if 'Videochess' not in x]
#
#     return lst


if __name__ == "__main__":
    main()
