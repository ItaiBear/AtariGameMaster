import gymnasium as gym
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
from wrappers import BinaryBoard, FrameSkipEnv

from evaluator import Evaluator

import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-E', type=str, default='TetrisA-v0')
    return parser.parse_args()

def make_env(args):
    env = gym.make(args.env, render_mode="human")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    #env = gym.wrappers.RecordVideo(env, "videos", step_trigger=lambda x: True)
    env = BinaryBoard(env)
    return env

def main():
    
    args = argparse.ArgumentParser()
    env = make_env(args)
    
    evaluator = Evaluator()
    try:
        obs, info = env.reset()
        done = False
        while not done:
            action 
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    
    
if __name__ == "__main__":
    main()