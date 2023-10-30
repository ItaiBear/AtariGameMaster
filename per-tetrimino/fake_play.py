import numpy as np

from internals.player import Player
from internals.globals import tetriminos
from internals.utils import format_board
    
def play_game():
    player = Player()
    total_reward = 0
    done = False
    while not done:
        print(f"current board: \n{format_board(player.tetris.board)}")
        tetrimino = np.random.choice(list(tetriminos.keys()))
        best_state, best_reward, done = player.find_best_state(tetrimino)
        if done:
            break
            
        total_reward += best_reward
        player.tetris.place_state(best_state)
        print(f"best state: {best_state.x}, {best_state.y}, {best_state.piece}, {best_state.orientation}")
        print(f"actions: {best_state.get_action_sequence()}")
            
    print(f"Total reward: {total_reward}")
    print(f"Total lines cleared: {player.tetris.total_lines_cleared}")
    return total_reward
    

def main():
    play_game()
    
if __name__ == "__main__":
    main()
        
        
            
        