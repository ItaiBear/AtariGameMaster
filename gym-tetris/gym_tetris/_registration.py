"""A script for registering environments with gym."""
import gym


# register for game mode A and B
for mode in {'A', 'B'}:
    b_type = mode == 'B'
    # v0: reward score
    gym.envs.registration.register(
        id='Tetris{}-v0'.format(mode),
        entry_point='gym_tetris:TetrisEnv',
        kwargs={
            'b_type': b_type,
            'reward_score': True,
            'reward_lines': False,
            'penalize_height': False,
        },
        nondeterministic=True,
    )
    # v1: reward lines
    gym.envs.registration.register(
        id='Tetris{}-v1'.format(mode),
        entry_point='gym_tetris:TetrisEnv',
        kwargs={
            'b_type': b_type,
            'reward_score': False,
            'reward_lines': True,
            'penalize_height': False,
        },
        nondeterministic=True,
    )
    # v2: reward score, penalize height
    gym.envs.registration.register(
        id='Tetris{}-v2'.format(mode),
        entry_point='gym_tetris:TetrisEnv',
        kwargs={
            'b_type': b_type,
            'reward_score': True,
            'reward_lines': False,
            'penalize_height': True,
        },
        nondeterministic=True,
    )
    # v3: reward lines, penalize height
    gym.envs.registration.register(
        id='Tetris{}-v3'.format(mode),
        entry_point='gym_tetris:TetrisEnv',
        kwargs={
            'b_type': b_type,
            'reward_score': False,
            'reward_lines': True,
            'penalize_height': True,
        },
        nondeterministic=True,
    )
    # v4: reward custom cost, score and bumpiness
    gym.envs.registration.register(
        id='Tetris{}-v4'.format(mode),
        entry_point='gym_tetris:TetrisEnv',
        kwargs={
            'b_type': b_type,
            'reward_score': True,
            'reward_lines': False,
            'penalize_height': False,
            'penalize_cost': True,
            'penalize_bumpiness': True
        },
        nondeterministic=True,
    )
    # v5: reward custom cost, score
    gym.envs.registration.register(
        id='Tetris{}-v5'.format(mode),
        entry_point='gym_tetris:TetrisEnv',
        kwargs={
            'b_type': b_type,
            'reward_score': True,
            'reward_lines': False,
            'penalize_height': False,
            'penalize_cost': True,
        },
        nondeterministic=True,
    )
    # v6: reward score, cumulative height, holes, bumpiness, penalize height
    gym.envs.registration.register(
        id='Tetris{}-v6'.format(mode),
        entry_point='gym_tetris:TetrisEnv',
        kwargs={
            'b_type': b_type,
            'reward_score': True,
            'reward_lines': False,
            'penalize_height': True,
            'penalize_cost': False,
            'penalize_holes': True,
            'penalize_bumpiness': True,
        },
        nondeterministic=True,
    )
    # v7:
    gym.envs.registration.register(
        id='Tetris{}-v7'.format(mode),
        entry_point='gym_tetris:TetrisEnv',
        kwargs={
            'b_type': b_type,
            'reward_score': True,
            'reward_lines': False,
            'penalize_height': True,
            'penalize_cost': False,
            'penalize_holes': False,
            'penalize_bumpiness': True,
            'penalize_transitions': True,
        },
        nondeterministic=True,
    )
    


# create an alias to gym.make for ease of access
make = gym.make


# define the outward facing API of this module (none, gym provides the API)
__all__ = [make.__name__]
