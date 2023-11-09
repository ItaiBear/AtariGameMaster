import minari
import argparse
import numpy as np
from collections import defaultdict
import pickle


append_to_dataset = False       # if true, load a predefined dataset and append to it


convert_to_binary = False       # if true, convert the observations from 84x84 grayscale board to binary
only_active_actions = False     # if true, only keep transitions which don't have a noop action
only_changing_states = False    # if true, only keep transitions with some change in the observation
remove_framestack = True        # if true, remove the framestack dimension
frame_stack = 4                 # number of frames to stack together (restack after removing framestack dimension)



def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='TetrisA-v0-itai-v0')
    parser.add_argument('--convert_to_binary', '-b', action='store_true', default=False)
    parser.add_argument('--only_active_actions', '-a', action='store_true', default=False)
    parser.add_argument('--only_changing_states', '-c', action='store_true', default=False)
    parser.add_argument('--remove_framestack', '-f', action='store_true', default=False)
    parser.add_argument('--frame_stack', '-s', type=int, default=4)
    return parser.parse_args()


args = argparser()
if args.dataset not in minari.list_local_datasets():
    raise ValueError('Dataset {} not found'.format(args.dataset))

dataset = minari.load_dataset(args.dataset)


if append_to_dataset:
    # load the dataset
    path = f'converted_datasets/TetrisA-v0-itai-v23_4.pkl'
    if not path.endswith("pkl"):
        path += "_1.pkl"
    with open(path, 'rb') as file:
        expert_trajs = pickle.load(file)
else:
    expert_trajs = defaultdict(list)
    
for episode in dataset.iterate_episodes():
    print(f'EPISODE ID {episode.id}')
    states = episode.observations
    actions = episode.actions
    rewards = episode.rewards
    terminations = episode.terminations
    truncations = episode.truncations
    dones = np.logical_or(terminations, truncations)
    length = episode.total_timesteps
    print("states shape: ", states.shape)
    print("actions shape: ", actions.shape)
    print("length: ", length)

    
    if convert_to_binary:
        BOARD_SHAPE = 20, 10
        y_step = 84 // BOARD_SHAPE[0]
        x_step = 84 // BOARD_SHAPE[1]
        cropped = states[:, :, y_step-1 : 84 - y_step + 1: y_step, (x_step//2) : 84 : x_step]
        assert cropped[0, 0].shape == BOARD_SHAPE, cropped[0, 0].shape
        cropped[cropped > 1] = 1.0
        cropped[cropped != 1] = 0.0
        states = cropped.astype(np.float32)
        
    if remove_framestack:
        states = states[:, -1, np.newaxis, :, :]
        print("removed framestack states shape: ", states.shape)
        
    if only_changing_states:
        # only keep states that are different from the next state
        changing_states_mask = np.full(states.shape[0], False)
        changing_states_mask[:-1] = np.any(states[1:] != states[:-1], axis=(1, 2, 3))
        changing_states_mask[-1] = True # final next state
        states = states[changing_states_mask]
        # set the last True to False (last state is the next state of the true last state)
        #changing_states_mask[np.nonzero(changing_states_mask)[0][-1]] = False
        changing_states_mask = changing_states_mask[:-1]
        actions = actions[changing_states_mask]
        rewards = rewards[changing_states_mask]
        dones = dones[changing_states_mask]
        length = np.sum(changing_states_mask)
    
        print("only changing states shape: ", states.shape)

    
    if frame_stack > 1:
        stacked_arr = np.empty((states.shape[0], frame_stack, *states.shape[-2:]))

        for i in range(states.shape[0]):
            # Get the frames to stack.
            frames_to_stack = np.squeeze(states[max(0, i-(frame_stack-1)):i+1])
            
            num_frames = frames_to_stack.shape[0]
            
            # Stack the frames in reverse order (so the current frame is first).
            stacked_arr[i, :num_frames] = frames_to_stack[::-1]  
            
            # If there are less than num_frames_to_stack frames, duplicate the earliest frame.
            if num_frames < frame_stack:
                stacked_arr[i, num_frames:] = np.squeeze(np.tile(frames_to_stack[-1], (frame_stack-num_frames, 1, 1, 1)))
        states = stacked_arr
        print("framestack states shape: ", states.shape)

    if only_active_actions:
        active_steps = actions != 0
    else:
        active_steps = np.ones_like(actions, dtype=bool)

    expert_trajs["states"].append(states[:-1][active_steps])
    expert_trajs["next_states"].append(states[1:][active_steps])
    expert_trajs["actions"].append(actions[active_steps])
    expert_trajs["rewards"].append(rewards[active_steps])

    expert_trajs["dones"].append(dones[active_steps])
    expert_trajs["lengths"].append(np.sum(active_steps))
    print("final episode length: ", np.sum(active_steps))


print('Final size of Replay Buffer: {}'.format(sum(expert_trajs["lengths"])))
with open(f'converted_datasets/{args.dataset}_{len(expert_trajs["lengths"])}.pkl', 'wb') as f:
    pickle.dump(expert_trajs, f)
exit()   