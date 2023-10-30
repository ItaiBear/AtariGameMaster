from .globals import *
from .utils import format_board
import numpy as np

class Tetris:
    def __init__(self, board):
        self.board = board
        self.height: int = board.shape[0]
        self.width: int = board.shape[1]
        self.total_lines_cleared = 0
        
    def is_valid_state(self, state):
        for dx, dy in orientations[state.orientation]:
            #print(f"dx: {dx}, dy: {dy}")
            x, y = state.x + dx, state.y + dy
            if x < 0 or x >= self.width or y < 0 or y >= self.height: # Out of bounds
                #print(f"Out of bounds at {x}, {y}")
                return False
            if self.board[y, x] == 1: # Occupied
                #print(f"Occupied at {x}, {y}")
                return False
        #print(f"Valid state: {state.x}, {state.y}, {state.piece}, {state.orientation}")
        return True
    
    def place_state(self, state) -> None:
        for dx, dy in orientations[state.orientation]:
            x, y = state.x + dx, state.y + dy
            self.board[y, x] = 1
        self.board, lines_cleared = self.clear_full_rows(self.board)
        self.total_lines_cleared += lines_cleared
        self.visited_states = np.zeros((self.width, self.height, len(tetriminos[state.piece])), dtype=bool)
        
    def get_updated_board(self, state) -> (np.ndarray, int, int):
        board = self.board.copy()
        #print(f"board before: \n{format_board(board)}")
        highest_y = 0
        for dx, dy in orientations[state.orientation]:
            x, y = state.x + dx, state.y + dy
            board[y, x] = 1
            if y > highest_y:
                highest_y = y
        board, lines_cleared = self.clear_full_rows(board)
        return board, lines_cleared, self.height - highest_y
    
    def clear_full_rows(self, board) -> (np.ndarray, int):
        # clear full rows
        lines_cleared = 0
        uncleared_board = board.copy()
        for row in range(self.height):
            if row == 0 and np.all(board[row, :] == 1):
                board = np.concatenate([np.zeros((1, self.width), dtype=np.int8), board[row+1:, :]], axis=0)
                lines_cleared += 1
            elif row == self.height-1 and np.all(board[row, :] == 1):
                board = np.concatenate([np.zeros((1, self.width), dtype=np.int8), board[:row, :]], axis=0)
                lines_cleared += 1
            elif np.all(board[row, :] == 1):
                board = np.concatenate([np.zeros((1, self.width), dtype=np.int8), board[:row, :], board[row+1:, :]], axis=0)
                lines_cleared += 1
        assert lines_cleared <= 4, f"Cleared {lines_cleared} lines, which is impossible"
        #if lines_cleared > 0:
            #print(f"board before clearing lines: \n{format_board(uncleared_board)}")
            #print(f"Cleared {lines_cleared} lines") 
        return board, lines_cleared