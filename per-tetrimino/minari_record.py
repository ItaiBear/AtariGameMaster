import gymnasium as gym
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
from wrappers import BinaryBoard, FrameSkipEnv, ExpandDim, FrameStack

from internals.player import Player
from internals.tetris import Tetris
from internals.state import State

import argparse
import numpy as np
import minari
import random


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-E', type=str, default='TetrisA-v0')
    parser.add_argument('--name', '-n', type=str, default='TetrisAfast-v0-itai-v0')
    parser.add_argument('--episodes', '-e', type=int, default=1)
    parser.add_argument('--seed', '-s', type=int, default=None)
    return parser.parse_args()

def make_env(args, framestack=16):
    env = gym.make(args.env, render_mode="human")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    #env = gym.wrappers.RecordVideo(env, "videos", step_trigger=lambda x: True)
    env = BinaryBoard(env)
    env = ExpandDim(env)
    env = FrameStack(env, framestack)

    env = minari.DataCollectorV0(env, record_infos=True, max_buffer_steps=100000)
    return env


MODIFIED_MOVEMENT = ['noop', 'clockwise', 'counterclockwise', 'right', 'left', 'down']

def play_episode(env, arg):
    player = Player()
    seed = random.randint(0, 2**32 - 1) if arg.seed is None else arg.seed

        
    total_reward = 0
    total_score = 0
    total_lines = 0
    total_pieces = 0
    total_actions = 0
    frame_counter = 0
    done = False
    try:
        obs, info = env.reset(seed=seed)
        obs, reward, terminated, truncated, info = env.step(MODIFIED_MOVEMENT.index('down'))
        total_actions += 1
        first_piece = True

        
        while not done:
            if first_piece:
                first_piece = False
            else:
                obs, reward, terminated, truncated, info = env.step(MODIFIED_MOVEMENT.index('noop'))
            total_actions += 1
            current_piece = info['current_piece'][0]
            player.set_level(info['level'])
            print(f"current piece: {current_piece}")
            print(f"level: {info['level']}")
            print(f"fall timer: {info['fall_timer']}")
            
            best_state, best_reward, should_be_done = player.find_best_state(current_piece, info['fall_timer'])
            actions = best_state.get_action_sequence()
            states = best_state.get_state_sequence()

            total_actions += len(actions)
            print(f"actions: {actions}")
            
            
            
            while actions:
                action = actions.pop(0)
                action_idx = MODIFIED_MOVEMENT.index(action)
                obs, reward, terminated, truncated, info = env.step(action_idx)
                expected_state = states.pop(0)
                print(f"fall timer: {info['fall_timer']}, state fall timer: {expected_state.fall_timer}, action: {action}, ai_dropped: {expected_state.drop}, auto_repeat: {expected_state.auto_repeat}, x: {expected_state.x}, y: {expected_state.y}")
                #time.sleep(5)

                #print(f"board: \n{states[0].board}")
                total_score = info['score']
                total_lines = info['number_of_lines']
                
                if info['is_piece_placed'] and actions:
                    print("piece placed before finishing action sequence")
                
            placed_piece = info['current_piece'][0]
                
            if info['is_piece_placed']:
                print("piece placed successfully")
            while not info['is_piece_placed']:
                print("piece not placed after finishing action sequence")
                action_idx = MODIFIED_MOVEMENT.index('down')
                obs, reward, terminated, truncated, info = env.step(action_idx)
                total_actions += 1
                total_score = info['score']
                total_lines = info['number_of_lines']
                
            while info['is_piece_placed']:
                action_idx = MODIFIED_MOVEMENT.index('noop')
                obs, reward, terminated, truncated, info = env.step(action_idx)
                total_actions += 1
                total_score = info['score']
                total_lines = info['number_of_lines']
                
            done = terminated or truncated
            
            fall_timer = info['fall_timer']
            info['fall_timer'] = 0

             
            total_reward += best_reward
            total_pieces += 1
            
            player.tetris.place_state(best_state)
            
            assert should_be_done == done, f"should_be_done: {should_be_done}, done: {done}"
            assert placed_piece == current_piece, f"placed_piece: {placed_piece}, current_piece: {current_piece}"
            #assert fall_timer == best_state.fall_timer, f"fall_timer: {fall_timer}, best_state.fall_timer: {best_state.fall_timer}"
            print(f"game fall timer: {fall_timer}, best_state.fall_timer: {best_state.fall_timer}")
            board = info['board']
            board[board == 239] = 0
            board[board != 0] = 1
            #assert info['is_piece_placed'], f"piece not placed, player board: \n{info['board']}"
            assert np.array_equal(board,player.tetris.board), f"board: \n{board}\nplayer board: \n{player.tetris.board}"
            
    except KeyboardInterrupt:
        pass
    finally:
        print(f"Total reward: {total_reward}")
        print(f"Total score: {total_score}")
        print(f"Total lines cleared: {total_lines}")
        print(f"Total placed tetriminos: {total_pieces}")
        print(f"Total actions played (frames): {total_actions}")
        


def main():
    args = argparser()
    env = make_env(args)
    checkpoint_interval = 1
    dataset_name = args.name
    dataset = None
    
    local_datasets = minari.list_local_datasets()
    if dataset_name in local_datasets:
        dataset = minari.load_dataset(dataset_name)
    #if dataset_name_h in local_datasets:
    #    dataset_h = minari.load_dataset(dataset_name_h)
        

    try:
        for episode_id in range(args.episodes):
            print(f"Episode {episode_id + 1}")
            try:
                play_episode(env, args)
            except:
                pass
            finally:
                if (episode_id + 1) % checkpoint_interval == 0:
                    if dataset is None:
                        dataset = minari.create_dataset_from_collector_env(dataset_id=dataset_name, 
                                                                            collector_env=env,
                                                                            algorithm_name="Expert-Demonstration",
                                                                            author="Itai Bear",
                                                                            author_email="itai.bear1@gmail.com")
                    else:
                        dataset.update_dataset_from_collector_env(env)
                    
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

        
        

    
if __name__ == "__main__":
    main()