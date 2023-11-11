import gym
import numpy as np


# Given an image of the current board, obtain a binary (20x10) representation
class BinaryBoard(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.GAME_BOX = 47, 95, 209, 176
        self.BOARD_SHAPE = 20, 10
        self.y_step = (self.GAME_BOX[2] - self.GAME_BOX[0]) // self.BOARD_SHAPE[0]
        self.x_step = (self.GAME_BOX[3] - self.GAME_BOX[1]) // self.BOARD_SHAPE[1]
        self.observation_space = gym.spaces.Box(0, 1, self.BOARD_SHAPE)

    def observation(self, obs):
        gray = np.mean(obs, axis=-1)
        cropped = gray[self.GAME_BOX[0]+(self.y_step//2) : self.GAME_BOX[2] : self.y_step,
                       self.GAME_BOX[1]+(self.x_step//2) : self.GAME_BOX[3] : self.x_step]
        assert cropped.shape == self.BOARD_SHAPE
        cropped[cropped > 1] = 1
        return cropped
    
class FrameSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(FrameSkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            # Only do the action on the first frame (action 0 is always NOOP)
            real_action = 0 if (i > 0) else action
            obs, reward, done, info = self.env.step(real_action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
    
class ExpandDim(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 1, (1,) + self.env.observation_space.shape)

    def observation(self, obs):
        return np.expand_dims(obs, axis=0)