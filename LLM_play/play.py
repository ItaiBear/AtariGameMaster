import openai
openai.organization = "org-nhnf3H43L8cLcRqXMZpfUL53"
openai.api_key = "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" # tetris openai API key

import gymnasium as gym
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
from wrappers import BinaryBoard, FrameSkipEnv

import argparse

MODEL = "gpt-3.5-turbo"
MODEL = "gpt-4"

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-E', type=str, default='TetrisA-v0')
    return parser.parse_args()

def format_board(board):
    board_str = ""
    for row in board:
        for col in row:
            board_str += " " + str(int(col))
        board_str += "\n"
    return board_str  

def make_prompt(current_piece):
    # Each action corresponds to a single frame in the game.
    # Be aware that Tetris NES incorporates a Delayed Auto Shift which means that if you hold down the left or right,
    # the piece will move in that direction by one column every 16 frames.
    # A good technique to avoid this is to alternate between shift actions and 'noop'. (e.g. 'left', 'noop', 'left', 'noop', 'left', 'noop')
    # In fact, you are only allowed to use 'noop' if it comes between two shift actions.
    system_prompt = \
    f"""
    You are a professional Tetris player.
    Given a board representation with 1s and 0s, you must decide on a sequence of actions to place the piece in the best possible position.
    Each piece has a single-character name that resembles its shape: 'T', 'J', 'Z', 'O', 'S', 'L', 'I'.
    current piece is {current_piece}
    Available actions are 'clockwise', 'counterclockwise', 'left', 'right', and 'drop'.
    You may repeat actions by specifying a number after the action (for example, 'left 2' will move the piece left by 2 columns).
    A 'drop' action is a hard drop, which means that the piece will fall to the bottom of the board.
    Your response structure should be as follows:
    board: describe the board using words.
    position: describe the optimal position in which you wish to place the piece.
    decision: provide the sequence of actions to be played. Reply only with the actions separated by commas, do not include any commentary.
    """
    return system_prompt

messages = [{"role": "system", "content": "foo"}]
def prompt_action_sequence(system_prompt, board):
    global messages
    messages[0]["content"] = system_prompt
    if len(messages) >= 5:
        messages = messages[:1] + messages[3:]
    messages.append({"role": "user", "content": board})
    completion = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
    )
    response = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": response})
    total_tokens = completion["usage"]["total_tokens"]
    return response, total_tokens


def extract_actions_sequence(response):
    """Extracts the actions sequence as a list from the response string."""
    decision = response.split("decision:")[-1]
    decisions = decision.split(",")
    characters_to_strip = ' .,!?"\''
    actions = []
    counts = []
    for decision in decisions:
        decision = decision.strip(characters_to_strip)
        action, *count = decision.split(" ")
        if len(count) == 0:
            count = 1
        else:
            count = count[0]
        print(f"action: {action}, count: {count}")
        actions.append(action)
        counts.append(count)
    action_sequence = generate_action_sequence(actions, counts)
    #actions = [s for s in actions if s != '']
    return action_sequence

def generate_action_sequence(actions, counts):
    assert len(actions) == len(counts), "actions and counts must be the same length"
    i = 1
    while i < len(actions):
        if actions[i] == actions[i-1]:
            counts[i] = int(counts[i]) + int(counts[i-1])
            actions.pop(i-1)
            counts.pop(i-1)
        else:
            i += 1
    sequence = []
    for action, count in zip(actions, counts):
        if action == 'left' or action == 'right':
            sequence += [action, 'noop'] * (int(count) - 1) + [action]
        elif action == 'clockwise' or action == 'counterclockwise':
            sequence += [action] * (int(count) % 4)
        elif action == 'down':
            sequence += [action] * (int(count) + 2)
        elif action == 'drop':
            sequence += ['down'] * 100
        else:
            print(f"Invalid action: {action} with count: {count}, skipping")
    return sequence
            

def make_env(args):
    env = gym.make(args.env, render_mode="human")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    #env = gym.wrappers.RecordVideo(env, "videos", step_trigger=lambda x: True)
    env = BinaryBoard(env)
    #env = FrameSkipEnv(env, 2)
    return env




SIMPLE_MOVEMENT = [
    ['NOOP'],
    ['A'],
    ['B'],
    ['right'],
    ['left'],
    ['down'],
]

MODIFIED_MOVEMENT = ['noop', 'counterclockwise', 'clockwise', 'right', 'left', 'down']

def main():
    TOTAL_TOKENS = 0
    args = argparser()
    env = make_env(args)
    try: 
        obs, info = env.reset()
        current_piece = info['current_piece'][0]

        total_reward = 0
        done = False
        new_piece = True
        actions = []
        while not done:
            if new_piece:
                action = 'noop'
                actions = ['noop']
            else:
                if len(actions) == 0:
                    board_str = format_board(obs)
                    system_prompt = make_prompt(current_piece)
                    print(f"board: \n{board_str}")
                    response, tokens = prompt_action_sequence(system_prompt, board_str)
                    print(f"response: {response}") 
                    TOTAL_TOKENS += tokens
                    actions = extract_actions_sequence(response)
                    print("actions: ", actions)
                
                action = actions.pop(0)

                
            try:
                action_idx = MODIFIED_MOVEMENT.index(action)
            except ValueError:
                print(f"Invalid action: {action}, using noop instead")
                action_idx = 0
            obs, reward, terminated, truncated, info = env.step(action_idx)
            new_piece = info['is_piece_placed']
            current_piece = info['current_piece'][0]
            #print(f"is piece placed: {new_piece}")
            total_reward += reward
            done = terminated or truncated
            
            
    finally:
        env.close()
        print(f"total reward: {total_reward}")
        print(f"total tokens: {TOTAL_TOKENS}")

if __name__ == "__main__":
    main()

