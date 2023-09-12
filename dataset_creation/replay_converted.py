import argparse
import pickle
import numpy as np
import cv2

def argparser():
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', '-d', type=str, default='TetrisA-v0')
    args.add_argument('--scale', '-s', type=int, default=10)
    return args.parse_args()

def main():
    args = argparser()
    path = args.dataset
    if not path.endswith(".pkl"):
        path += ".pkl"
    with open(path, 'rb') as file:
        dataset = pickle.load(file)
        
    print("dataset keys: ", dataset.keys())
    print("total dataset steps: ", np.sum(dataset["lengths"]))
    print("State observation shape: ", dataset["states"][0][0].shape)
    for i in range(len(dataset["lengths"])): # iterate over episodes
        print("episode ", i)
        print("episode length: ", dataset["lengths"][i])
        print("first 100 actions: ", dataset["actions"][i][:100])
        episode_states = dataset["states"][i]
        for state in episode_states:
            frame = state[-1]
            scaled_frame = np.repeat(np.repeat(frame, args.scale, axis=0), args.scale, axis=1)
            cv2.imshow("state", scaled_frame)
            # wait for 1ms
            cv2.waitKey(1)
            
if __name__ == "__main__":
    main()
