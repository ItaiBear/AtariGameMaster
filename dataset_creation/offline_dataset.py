from nes_py.wrappers import JoypadSpace
import gym_tetris

import gymnasium as gym
import numpy as np
import minari

from gym_tetris.actions import SIMPLE_MOVEMENT

import time
from pyglet import clock
from nes_py._image_viewer import ImageViewer
from wrappers import *

import argparse
import random

_NOP = 0


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-E', type=str, default='TetrisA-v0')
    parser.add_argument('--name', '-n', type=str, default='TetrisAfast-v0-itai-v0')
    parser.add_argument('--episodes', '-e', type=int, default=1)
    parser.add_argument('--seed', '-s', type=int, default=1)
    return parser.parse_args()

args = argparser()

SCALE = 2

INNER_SKIP = 1
OUTER_SKIP = 1
framestack = 4

env_h = gym.make(args.env, render_mode=None)
env_h = JoypadSpace(env_h, SIMPLE_MOVEMENT)
env_h = minari.DataCollectorV0(env_h, record_infos=True, max_buffer_steps=100000)

env = gym.make(args.env, render_mode=None)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = BinaryBoard(env)
env = FrameSkipEnv(env, INNER_SKIP)
env = ExpandDim(env)
env = FrameStack(env, framestack)

env = minari.DataCollectorV0(env, record_infos=True, max_buffer_steps=100000)


checkpoint_interval = 1
dataset_name = args.name
dataset_name_split = args.name.split("-")
dataset_name_split[-2] += "_h"
dataset_name_h = "-".join(dataset_name_split)
dataset = None
dataset_h = None

local_datasets = minari.list_local_datasets()
if dataset_name in local_datasets:
    dataset = minari.load_dataset(dataset_name)
#if dataset_name_h in local_datasets:
#    dataset_h = minari.load_dataset(dataset_name_h)
    
seed = random.randint(0, 2**32 - 1) if args.seed is None else args.seed
    


# get the mapping of keyboard keys to actions in the environment
if hasattr(env_h, 'get_keys_to_action'):
    keys_to_action = env_h.get_keys_to_action()
elif hasattr(env_h.unwrapped, 'get_keys_to_action'):
    keys_to_action = env_h.unwrapped.get_keys_to_action()
else:
    raise ValueError('env_h has no get_keys_to_action method')
# create the image viewer
viewer = ImageViewer(
    env_h.spec.id if env_h.spec is not None else env_h.__class__.__name__,
    env_h.observation_space.shape[0] * SCALE, # height
    env_h.observation_space.shape[1] * SCALE, # width
    monitor_keyboard=True,
    relevant_keys=set(sum(map(list, keys_to_action.keys()), []))
)

# prepare frame rate limiting
target_frame_duration = 1 / env_h.metadata['video.frames_per_second']
last_frame_time = 0

for episode_id in range(args.episodes):

    done = False
    state_h, _ = env_h.reset(seed=seed)
    state, _ = env.reset(seed=seed)
    viewer.show(env_h.unwrapped.screen)
    step = 0 

    # start the main game loop
    try:
        while not done:
            current_frame_time = time.time()
            # limit frame rate
            if last_frame_time + target_frame_duration > current_frame_time:
                continue
            # save frame beginning time for next refresh
            last_frame_time = current_frame_time
            
            clock.tick()
            
            # unwrap the action based on pressed relevant keys
            action = _NOP
            if step % INNER_SKIP == 0:
                action = keys_to_action.get(viewer.pressed_keys, _NOP)
                next_state, reward, terminated, truncated, _ = env.step(action)
            _, _, _, _, _ = env_h.step(action)
            done = terminated or truncated
            viewer.show(env_h.unwrapped.screen)
            step += 1
            # shutdown if the escape key is pressed
            if viewer.is_escape_pressed:
                break
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        pass
    except AttributeError:
        print("AttributeError")
        pass
        
    if (episode_id + 1) % checkpoint_interval == 0:
        if dataset is None:
            dataset = minari.create_dataset_from_collector_env(dataset_id=dataset_name, 
                                                        collector_env=env,
                                                        algorithm_name="Human-Demonstration",
                                                        author="Itai Bear",
                                                        author_email="itai.bear1@gmail.com")
            
        else:
            dataset.update_dataset_from_collector_env(env)
            
        # if dataset_h is None:
        #     dataset_h = minari.create_dataset_from_collector_env(dataset_id=dataset_name_h,
        #                                                          collector_env=env_h,
        #                                                          algorithm_name="Human-Demonstration",
        #                                                          author="Itai Bear",
        #                                                          author_email="itai.bear1@gmail.com")
        # else:
        #     dataset_h.update_dataset_from_collector_env(env_h)
            
viewer.close()            
env.close()
env_h.close()