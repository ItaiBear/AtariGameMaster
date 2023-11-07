"""
Script for training an RL agent to play in the Per-Tetrimino Gameplay of Tetris.
Trains on a simulation that mimicks the Tetris NES environment. Reward is based on the evaluator function.
"""

import random
from models import TetrisNetwork
import torch
import torch.optim as optim
import torch.nn.functional as F
from internals.player import Player
from internals.globals import tetriminos
from internals.evaluator import Evaluator
from internals.rl_utils import linear_schedule, update_target_model
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

import time
import os
from tqdm import tqdm

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
    

def train():
    input_dim = (20, 10)
    
    # Load the config YAML file
    args = OmegaConf.load(os.path.join('configs', 'train_simulated_rl.yaml'))
    run_name = f"tetris__{args.exp_name}__{args.seed}__{int(time.time())}"
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=OmegaConf.to_container(args, resolve=True),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    #np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(args.device_name)
    print("device_name:", args.device_name)
    
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

    evaluator = Evaluator()
    # Training loop
    for episode in tqdm(range(args.total_episodes)):  # Number of episodes
        player = Player()
        episodic_reward = 0
        episodic_length = 0
        episodic_lines = 0
        done = False
        while True:
            global_step += 1
            episodic_length += 1
            tetrimino = np.random.choice(list(tetriminos.keys()))
            
            observation = player.get_current_board()
            locked_states = player.bfs(tetrimino)
            
            if not locked_states:   # episode is over
                tqdm.write(f"global_step={global_step}, episodic_return={episodic_reward}, episodic_length={episodic_length}")
                writer.add_scalar("charts/episodic_return", episodic_reward, episode)
                writer.add_scalar("charts/episodic_length", episodic_length, episode)
                writer.add_scalar("charts/episodic_pieces", episodic_length, episode)
                writer.add_scalar("charts/episodic_lines", episodic_lines, episode)
                writer.add_scalar("charts/epsilon", epsilon, episode)
                break
            
            num_states = len(locked_states)
            boards = np.empty((num_states, *input_dim))
            cleared_lines = np.empty(num_states, dtype=int)
            piece_height = np.empty(num_states, dtype=int)

            for i, state in enumerate(locked_states):
                board, lines, height = player.tetris.get_updated_board(state)
                boards[i] = board
                cleared_lines[i] = lines
                piece_height[i] = height
            
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_episodes, episode)

            best_state_idx = select_state_idx(network, boards, epsilon, device)
            best_state = locked_states[best_state_idx]
            episodic_lines += cleared_lines[best_state_idx]
            player.tetris.place_state(best_state)
            reward = evaluator.evaluate(player.tetris.board, cleared_lines[best_state_idx], piece_height[best_state_idx])
            reward = reward / 10000
                
            episodic_reward += reward
            replay_buffer.append((observation, player.get_current_board(), reward, done))
            
            if len(replay_buffer) > args.learning_starts:
                if len(replay_buffer) == args.learning_starts + 1:
                    print("learning starts!")
                if global_step % args.train_frequency == 0:
                    # Sample mini-batch from replay buffer
                    mini_batch = random.choices(replay_buffer, k=args.batch_size)
                    observations, next_observations, rewards, dones = zip(*mini_batch)
                    
                    #observations = np.stack(observations)

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
                
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        
    writer.close()
                
        
def main():
    train()

if __name__ == '__main__':
    main()