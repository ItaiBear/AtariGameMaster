# Environement Configuration
class FrameSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4, only_first=False):
        super(FrameSkipEnv, self).__init__(env)
        self._skip = skip
        self._only_first = only_first

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            # Only do the action on the first frame (action 0 is always NOOP)
            real_action = 0 if (self._only_first and i > 0) else action
            obs, reward, done, info = self.env.step(real_action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        return obs

BOX = 47, 95, 209, 176
# Making an environment
def get_env(env_id, seed, capture_video, run_name, video_freq=100, frame_stack=4):
    env = gym_tetris.make(env_id)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if capture_video:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda ep_num: ep_num % video_freq == 0)
    
    crop = lambda obs : obs[BOX[0]:BOX[2], BOX[1]:BOX[3], :]
    env = gym.wrappers.TransformObservation(env, crop)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)

    env = FrameSkipEnv(env, skip=16, only_first=True)
    env = gym.wrappers.FrameStack(env, frame_stack)
    env = FrameSkipEnv(env, skip=2, only_first=False)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


# Training Arguments
class Args:
    def __init__(self):
        # Settings
        self.exp_name = "Tetris_DQN"
        self.torch_deterministic = True
        self.cuda = True
        self.mps = False
        self.capture_video = True
        self.capture_inputs_video = True
        self.save_model = True
        self.eval_episodes = 1
        self.video_frequency = 50
        self.backup_frequency = 10000

        # Hyper-Parameters
        self.env_id = "TetrisA-v5"
        self.frame_stack = 4
        self.seed = 2
        self.total_timesteps = 1_000_000
        self.learning_rate = 1e-4
        self.buffer_size = 50_000
        self.gamma = 0.99
        self.tau = 0.999
        self.target_network_frequency = 2000
        self.batch_size = 32
        self.start_e = 1
        self.end_e = 0.05
        self.exploration_fraction = 0.2
        self.learning_starts = 40_000
        self.train_frequency = 1
