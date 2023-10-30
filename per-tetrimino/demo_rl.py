import gymnasium as gym
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
from wrappers import BinaryBoard, FrameSkipEnv, ExpandDim, FrameStack

from internals.player import Player
from models import TetrisNetwork

import argparse
import numpy as np
import torch


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-E', type=str, default='TetrisA-v0')
    parser.add_argument('--seed', '-s', type=int, default=1)
    # if record is given, set it to true, otherwise set it to false
    parser.add_argument('--record', '-r', action='store_true', default=False)
    parser.add_argument('--model_path', '-m', type=str, default=None)
    parser.add_argument('--device', '-d', type=str, default='cpu')
    return parser.parse_args()

def make_env(args):
    render_mode = "rgb_array" if args.record else "human"
    env = gym.make(args.env, render_mode=render_mode)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    if args.record:
        env = gym.wrappers.RecordVideo(env, "videos", step_trigger=lambda x: True)
    env = BinaryBoard(env)
    return env


MODIFIED_MOVEMENT = ['noop', 'clockwise', 'counterclockwise', 'right', 'left', 'down']


def main():
    args = argparser()
    device = torch.device(args.device)
    env = make_env(args)
    
    player = Player()
    
    #model_path = "runs/tetris__test__42__1694433043/test_35000.backup"
    #model_path = "runs/tetris__test__42__1694392640/test_210000.backup"
    #model_path = "runs/tetris__test__42__1694439794/test_10000.backup"
    #model_path = "runs/tetris__test__42__1694445084/test.cleanrl_model"
    #model_path = "runs/tetris__test__42__1694449993/test.cleanrl_model"
    model_path = "runs/tetris__test__42__1694519568/checkpoints/test_620000.backup"
    network = TetrisNetwork((20, 10)).to(device)
    network.load_state_dict(torch.load(model_path, map_location=device))
    network.eval()
    
    total_reward = 0
    total_score = 0
    total_lines = 0
    total_pieces = 0
    total_actions = 0
    frame_counter = 0
    done = False
    try:
        obs, info = env.reset(seed=args.seed)
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
            
            #best_state, best_reward, should_be_done = player.find_best_state(current_piece, info['fall_timer'])
            locked_states = player.bfs(current_piece, info['fall_timer'])
            if not locked_states:
                print("no locked states")
                break
            
            updated_locked_states = [player.tetris.get_updated_board(state) for state in locked_states]
            boards, cleared_lines, piece_height = [], [], []
            for state in updated_locked_states:
                boards.append(state[0])
                cleared_lines.append(state[1])
                piece_height.append(state[2])
                
            with torch.no_grad():
                stacked_boards = torch.Tensor(boards).unsqueeze(1).to(device)
                values = network(stacked_boards).squeeze()
                state_idx = torch.argmax(values).cpu().numpy()
                
            best_state = locked_states[state_idx]
            
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
                done = terminated or truncated
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
                done = terminated or truncated
                total_actions += 1
                total_score = info['score']
                total_lines = info['number_of_lines']
                
            while info['is_piece_placed']:
                action_idx = MODIFIED_MOVEMENT.index('noop')
                obs, reward, terminated, truncated, info = env.step(action_idx)
                done = terminated or truncated
                total_actions += 1
                total_score = info['score']
                total_lines = info['number_of_lines']
            
            fall_timer = info['fall_timer']
            info['fall_timer'] = 0

             
            #total_reward += best_reward
            total_pieces += 1
            
            player.tetris.place_state(best_state)
            
            #assert should_be_done == done, f"should_be_done: {should_be_done}, done: {done}"
            assert placed_piece == current_piece, f"placed_piece: {placed_piece}, current_piece: {current_piece}"
            #assert fall_timer == best_state.fall_timer, f"fall_timer: {fall_timer}, best_state.fall_timer: {best_state.fall_timer}"
            print(f"game fall timer: {fall_timer}, best_state.fall_timer: {best_state.fall_timer}")
            board = info['board']
            board[board == 239] = 0
            board[board != 0] = 1
            #assert info['is_piece_placed'], f"piece not placed, player board: \n{info['board']}"
            if not np.array_equal(board,player.tetris.board):
                print("board mismatch, updating player board with game board")
                player.tetris.board = board
                
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        #print(f"Total reward: {total_reward}")
        print(f"Total score: {total_score}")
        print(f"Total lines cleared: {total_lines}")
        print(f"Total placed tetriminos: {total_pieces}")
        print(f"Total actions played (frames): {total_actions}")
        
        

    
if __name__ == "__main__":
    main()