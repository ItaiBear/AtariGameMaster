import gymnasium as gym
import numpy as np
from collections import deque
from gymnasium import spaces
import torch

class CropObservation(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, BOX: tuple):
        gym.utils.RecordConstructorArgs.__init__(self, BOX=BOX)
        gym.ObservationWrapper.__init__(self, env)

        self.BOX = BOX

    def observation(self, observation):
        return observation[self.BOX[0]:self.BOX[2], self.BOX[1]:self.BOX[3], :]
    
class PyTorchFrame(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Image shape to num_channels x height x width"""

    def __init__(self, env):
        gym.utils.RecordConstructorArgs.__init__(self)
        super(PyTorchFrame, self).__init__(env)
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(
            shape[-1], shape[0], shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return np.rollaxis(observation, 2)
    
class FrameStack(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        Expects inputs to be of shape num_channels x height x width.
        """
        gym.utils.RecordConstructorArgs.__init__(self, k=k)
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(
            shp[0] * k, shp[1], shp[2]), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        ob, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, terminated, truncated, info
    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]
    


# Given an image of the current board, obtain a binary (20x10) representation
class BinaryBoard(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env):
        gym.utils.RecordConstructorArgs.__init__(self)
        super().__init__(env)
        self.GAME_BOX = 47, 95, 209, 176
        self.BOARD_SHAPE = 20, 10
        self.y_step = (self.GAME_BOX[2] - self.GAME_BOX[0]) // self.BOARD_SHAPE[0]
        self.x_step = (self.GAME_BOX[3] - self.GAME_BOX[1]) // self.BOARD_SHAPE[1]
        self.observation_space = spaces.Box(0, 1, self.BOARD_SHAPE)

    def observation(self, obs):
        gray = np.mean(obs, axis=-1)
        cropped = gray[self.GAME_BOX[0]+(self.y_step//2) : self.GAME_BOX[2] : self.y_step,
                       self.GAME_BOX[1]+(self.x_step//2) : self.GAME_BOX[3] : self.x_step]
        assert cropped.shape == self.BOARD_SHAPE
        cropped[cropped > 1] = 1
        return cropped
    
class FrameSkipEnv(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env, skip=4):
        gym.utils.RecordConstructorArgs.__init__(self, skip=skip)
        super(FrameSkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            # Only do the action on the first frame (action 0 is always NOOP)
            real_action = 0 if (i > 0) else action
            obs, reward, terminated, truncated, info = self.env.step(real_action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
    
class ExpandDim(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env):
        gym.utils.RecordConstructorArgs.__init__(self)
        super().__init__(env)
        self.observation_space = spaces.Box(0, 1, (1,) + self.env.observation_space.shape)

    def observation(self, obs):
        return np.expand_dims(obs, axis=0)
    
