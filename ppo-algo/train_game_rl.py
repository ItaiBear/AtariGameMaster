import random
from models import TetrisNetwork
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from internals.player import Player
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

import gymnasium as gym
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
from wrappers import BinaryBoard

import time
import os
from tqdm import tqdm


def make_env(args):
    render_mode = "rgb_array" if args.record_video else "human"
    env = gym.make(args.env, render_mode=render_mode)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    if args.record_video:
        env = gym.wrappers.RecordVideo(env, "videos", step_trigger=capped_cubic_video_schedule)
    env = BinaryBoard(env)
    return env

def capped_cubic_video_schedule(episode_id: int) -> bool:
    if episode_id < 100:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    return episode_id % 100 == 0

# Epsilon-greedy action selection
def select_state_idx(net, locked_states, epsilon, device=torch.device('cpu')) -> int:
    if random.random() < epsilon:
        return random.choice(range(len(locked_states)))
    else:
        with torch.no_grad():
            stacked_states = torch.Tensor(locked_states).unsqueeze(1).to(device)
            values = net(stacked_states).squeeze()
            state_idx = torch.argmax(values).cpu().numpy()
        return state_idx
    
# Function to sync target network
def update_target_model(net, target_net, tau=1.0):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1.0 - tau) * target_param.data
        )
        
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

MODIFIED_MOVEMENT = ['noop', 'clockwise', 'counterclockwise', 'right', 'left', 'down']

def train():
    input_dim = (20, 10)
    
    # Load the config YAML file
    args = OmegaConf.load(os.path.join('configs', 'train_game_rl.yaml'))
    run_name = f"realtetris__{args.exp_name}__{args.seed}__{int(time.time())}"
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=OmegaConf.to_container(args, resolve=True),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(args.device_name)
    print("device_name:", args.device_name)
    
    env = make_env(args)
    
    replay_buffer = deque(maxlen=args.buffer_size)

    # Initialize target network and replay buffer
    network = TetrisNetwork(input_dim).to(device)
    target_network = TetrisNetwork(input_dim).to(device)
    
    if args.pretrain:
        network.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))
    target_network.load_state_dict(network.state_dict())
    #network = torch.compile(network)
    #target_network = torch.compile(target_network)

    optimizer = optim.Adam(network.parameters(), lr=args.learning_rate)
    
    start_time = time.time()
    global_step = 0

    # Training loop
    try:
        for episode in tqdm(range(args.total_episodes)):  # Number of episodes
            player = Player()
            episodic_reward = 0
            episodic_length = 0
            total_pieces = 0
            total_actions = 0
            done = False

            obs, info = env.reset(seed=args.seed)
            first_piece = True
            while not done:
                decision_reward = 0
                if first_piece:
                    first_piece = False
                    obs, reward, terminated, truncated, info = env.step(MODIFIED_MOVEMENT.index('down'))
                else:
                    obs, reward, terminated, truncated, info = env.step(MODIFIED_MOVEMENT.index('noop'))
                decision_reward += reward
                total_actions += 1
                    
                current_piece = info['current_piece'][0]
                player.set_level(info['level'])

                observation = player.get_current_board()
                locked_states = player.bfs(current_piece, info['fall_timer'])
            
                if not locked_states:   # episode is over
                    tqdm.write(f"global_step={global_step}, episodic_return={episodic_reward}")
                    writer.add_scalar("charts/episodic_return", episodic_reward, episode)
                    writer.add_scalar("charts/episodic_length", episodic_length, episode)
                    writer.add_scalar("charts/episodic_score", info['score'], global_step)
                    writer.add_scalar("charts/episodic_lines", info['number_of_lines'], global_step)
                    writer.add_scalar("charts/episodic_pieces", total_pieces, global_step)
                    writer.add_scalar("charts/episodic_actions", total_actions, global_step)
                    writer.add_scalar("charts/epsilon", epsilon, global_step)
                    break
                
                global_step += 1
                episodic_length += 1
            
                num_states = len(locked_states)
                boards = np.empty((num_states, *input_dim))
                cleared_lines = np.empty(num_states, dtype=int)
                piece_height = np.empty(num_states, dtype=int)
                for i, state in enumerate(locked_states):
                    board, lines, height = player.tetris.get_updated_board(state)
                    boards[i] = board
                    cleared_lines[i] = lines
                    piece_height[i] = height
            

                # choose a state
                epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_episodes, episode)
                best_state_idx = select_state_idx(network, boards, epsilon, device)
                best_state = locked_states[best_state_idx]
                player.tetris.place_state(best_state)
                
                # play according to state actions
                actions = best_state.get_action_sequence()
                #states = best_state.get_state_sequence()
                total_actions += len(actions)
                
                while actions:
                    action = actions.pop(0)
                    obs, reward, terminated, truncated, info = env.step(MODIFIED_MOVEMENT.index(action))
                    decision_reward += reward
                    #expected_state = states.pop(0)
                    #tqdm.write(f"fall timer: {info['fall_timer']}, state fall timer: {expected_state.fall_timer}, action: {action}, ai_dropped: {expected_state.drop}, auto_repeat: {expected_state.auto_repeat}, x: {expected_state.x}, y: {expected_state.y}")
                    
                    #if info['is_piece_placed'] and actions:
                        #tqdm.write("piece placed before finishing action sequence")
                    
                placed_piece = info['current_piece'][0]
                    
                #if info['is_piece_placed']:
                    #tqdm.write("piece placed successfully")
                while not info['is_piece_placed']: # piece has not been locked, keep moving down
                    #tqdm.write("piece not placed after finishing action sequence")
                    obs, reward, terminated, truncated, info = env.step(MODIFIED_MOVEMENT.index('down'))
                    decision_reward += reward
                    total_actions += 1
                    
                while info['is_piece_placed']: # piece has been locked, advance to next piece
                    obs, reward, terminated, truncated, info = env.step(MODIFIED_MOVEMENT.index('noop'))
                    decision_reward += reward
                    total_actions += 1
                
                done = terminated or truncated
                
                fall_timer = info['fall_timer']
                info['fall_timer'] = 0
                
                total_pieces += 1
                #decision_reward = (decision_reward / 1000) - 3
                episodic_reward += decision_reward
                
                replay_buffer.append((observation, player.get_current_board(), decision_reward, done))
                
                if len(replay_buffer) > args.learning_starts:
                    if len(replay_buffer) == args.learning_starts + 1:
                        print("learning starts!")
                    if global_step % args.train_frequency == 0:
                        # Sample mini-batch from replay buffer
                        mini_batch = random.choices(replay_buffer, k=args.batch_size)
                        observations, next_observations, rewards, dones = zip(*mini_batch)

                        observations = torch.Tensor(np.array(observations, dtype=np.float64)).squeeze(dim=0).unsqueeze(1).to(device)
                        rewards = torch.Tensor(np.array(rewards)).squeeze().to(device)
                        next_observations = torch.Tensor(np.array(next_observations, dtype=np.float64)).squeeze(dim=0).unsqueeze(1).to(device)
                        dones = torch.Tensor(np.array(dones)).squeeze().to(device)
                        #print(f"rewards flatten: {rewards.flatten()}")

                        with torch.no_grad():
                            # Forward pass to get next state values from target network
                            target_max, _ = target_network(next_observations).max(dim=1)
                            td_target = rewards.flatten() + args.gamma * target_max * (1 - dones.flatten())
                        
                        # Forward pass to get current state values
                        old_val = network(observations).squeeze()
                        loss = F.mse_loss(td_target, old_val)
                        
                        if global_step % 100 == 0:
                            writer.add_scalar("losses/td_loss", loss, global_step)
                            writer.add_scalar("losses/net_values", old_val.mean().item(), global_step)
                            #print("SPS:", int(global_step / (time.time() - start_time)))
                            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                        # Update model weights
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # Periodically update the target network
                    if global_step % args.target_network_update_frequency == 0:  # Choose a suitable frequency
                        update_target_model(network, target_network, args.tau)
                        
                    if args.save_model and global_step % args.backup_frequency == 0:
                        model_path = f"runs/{run_name}/checkpoints/{args.exp_name}_{global_step}.backup"
                        torch.save(network.state_dict(), model_path)
                        
                assert placed_piece == current_piece, f"placed_piece: {placed_piece}, current_piece: {current_piece}"
                #assert fall_timer == best_state.fall_timer, f"fall_timer: {fall_timer}, best_state.fall_timer: {best_state.fall_timer}"
                #print(f"game fall timer: {fall_timer}, best_state.fall_timer: {best_state.fall_timer}")
                board = np.where(info['board'] == 239, 0, 1)
                #assert info['is_piece_placed'], f"piece not placed, player board: \n{info['board']}"
                if not np.array_equal(board,player.tetris.board):
                    print("board mismatch, updating player board with game board")
                    player.tetris.board = board
                    
    except KeyboardInterrupt:
        pass
    
    finally:  
        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(network.state_dict(), model_path)
            print(f"model saved to {model_path}")
            
        writer.close()
        env.close()
        wandb.finish()
                
        
def main():
    train()

if __name__ == '__main__':
    main()