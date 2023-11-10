"""
Script to replay existing minari datasets
"""

import gymnasium as gym
import minari
import argparse
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
from nes_py.wrappers.joypad_space import JoypadSpace
from nes_py._image_viewer import ImageViewer
from wrappers import FrameSkipEnv
import time
from pyglet import clock
import numpy as np
import cv2

convert_to_binary = False


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-E', type=str, default='TetrisA-v0')
    parser.add_argument('dataset', type=str)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--skip', '-k', type=int, default=1)
    return parser.parse_args()

args = argparser()

env = gym.make(args.env, render_mode=None)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = FrameSkipEnv(env, args.skip)

SCALE = 2

# get the mapping of keyboard keys to actions in the environment
if hasattr(env, 'get_keys_to_action'):
    keys_to_action = env.get_keys_to_action()
elif hasattr(env.unwrapped, 'get_keys_to_action'):
    keys_to_action = env.unwrapped.get_keys_to_action()
else:
    raise ValueError('env_h has no get_keys_to_action method')
# create the image viewer
viewer = ImageViewer(
    env.spec.id if env.spec is not None else env.__class__.__name__,
    env.observation_space.shape[0] * SCALE, # height
    env.observation_space.shape[1] * SCALE, # width
    monitor_keyboard=False,
    relevant_keys=set(sum(map(list, keys_to_action.keys()), []))
)

observation_viewer = ImageViewer("binary_board", 200, 100)
    

# prepare frame rate limiting
FPS = env.metadata['video.frames_per_second']
target_frame_duration = 1 / FPS
last_frame_time = 0

print('obs space shape:', env.observation_space.shape)
print('action space:', env.action_space)


if args.dataset not in minari.list_local_datasets():
    raise ValueError('Dataset {} not found'.format(args.dataset))

dataset = minari.load_dataset(args.dataset)

# recovered_env = dataset.recover_environment()
# print('recovered env obs shape:', recovered_env.observation_space.shape)


for episode in dataset.iterate_episodes():
    print(f'EPISODE ID {episode.id}')
    actions = episode.actions
    observations = episode.observations
    print('original observations shape:', episode.observations.shape)
    if convert_to_binary:
        BOARD_SHAPE = 20, 10
        y_step = 84 // BOARD_SHAPE[0]
        x_step = 84 // BOARD_SHAPE[1]
        cropped = observations[:, :, y_step-1 : 84 - y_step + 1: y_step, (x_step//2) : 84 : x_step]
        assert cropped[0, 0].shape == BOARD_SHAPE, cropped[0, 0].shape
        cropped[cropped > 1] = 1.0
        cropped[cropped != 1] = 0.0
        observations = cropped.astype(np.float32)
    print('example cropped observation:', observations[0, 0])
    print('cropped observations shape:', observations.shape)
    observations = observations[:, -1]
    observations = np.expand_dims(observations, axis=-1)
    print('observations shape:', observations.shape)
    i = 0
    step = 0
    episodic_reward = 0
    terminated, truncated = False, False
    _, _ = env.reset(seed=int(episode.seed))
    viewer.show(env.unwrapped.screen)
    while not (terminated or truncated):
        current_frame_time = time.time()
        # limit frame rate
        if last_frame_time + target_frame_duration > current_frame_time:
            continue
        # save frame beginning time for next refresh
        last_frame_time = current_frame_time
        
        clock.tick()
        if step % args.skip == 0:
            action = actions[i]
            obs, rew, terminated, truncated, info = env.step(action)
            cv2.imshow('observation', np.repeat(np.repeat(observations[i], 10, axis=0), 10, axis=1))
            viewer.show(env.unwrapped.screen)
            episodic_reward += rew
            i += 1
        step += 1
    print(f'EPISODE REWARD {episodic_reward}')
env.close()

