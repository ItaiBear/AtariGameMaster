"""A script for registering environments with gymnasium."""
import gymnasium as gym

# register for game mode A and B
for mode in {'A', 'B'}:
    for level in {"", "fast"}:
        b_type = mode == 'B'
        level_9 = level == "fast"
        # v0: reward score
        gym.envs.registration.register(
            id='Tetris{}{}-v0'.format(mode, level),
            entry_point='gym_tetris:TetrisEnv',
            kwargs={
                'b_type': b_type,
                'level_9': level_9,
                'reward_score': True,
                'reward_lines': False,
                'penalize_height': False,
            },
            nondeterministic=True,
        )
        # v1: reward lines
        gym.envs.registration.register(
            id='Tetris{}{}-v1'.format(mode, level),
            entry_point='gym_tetris:TetrisEnv',
            kwargs={
                'b_type': b_type,
                'level_9': level_9,
                'reward_score': False,
                'reward_lines': True,
                'penalize_height': False,
            },
            nondeterministic=True,
        )
        # v2: reward score, penalize height
        gym.envs.registration.register(
            id='Tetris{}{}-v2'.format(mode, level),
            entry_point='gym_tetris:TetrisEnv',
            kwargs={
                'b_type': b_type,
                'level_9': level_9,
                'reward_score': True,
                'reward_lines': False,
                'penalize_height': True,
            },
            nondeterministic=True,
        )
        # v3: reward lines, penalize height
        gym.envs.registration.register(
            id='Tetris{}{}-v3'.format(mode, level),
            entry_point='gym_tetris:TetrisEnv',
            kwargs={
                'b_type': b_type,
                'level_9': level_9,
                'reward_score': False,
                'reward_lines': True,
                'penalize_height': True,
            },
            nondeterministic=True,
        )


# create an alias to gym.make for ease of access
make = gym.make


# define the outward facing API of this module (none, gym provides the API)
__all__ = [make.__name__]
