import gym
import sys
# import numpy as np

def main():
    print(sys.argv[0])

    env = gym.make(f'ALE/{sys.argv[1]}-v5', render_mode='human')

    obs = env.reset()

    done = False

    i = 0
    while not done:
        act = input('Choose next action: ')
        if act == '': act = 0
        act = int(act)
        obs, rew, done, truncated, info = env.step(act)
        env.render()
        i += 1

    env.close()




if __name__ == "__main__":
    main()
