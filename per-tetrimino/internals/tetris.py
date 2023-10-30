from .globals import *
from .utils import format_board
import numpy as np

class Tetris:
    '''
    Class representing the tetris board and its mechanics
    '''
    def __init__(self, board):
        self.board = board
        self.height: int = board.shape[0]
        self.width: int = board.shape[1]
        self.total_lines_cleared = 0
        
    def is_valid_state(self, state) -> bool:
        for dx, dy in orientations[state.orientation]:
            x, y = state.x + dx, state.y + dy
            if x < 0 or x >= self.width or y < 0 or y >= self.height: # Out of bounds
                return False
            if self.board[y, x] == 1: # Occupied cell
                return False
        return True
    
    def place_state(self, state) -> None:
        # Places the state on the board
        for dx, dy in orientations[state.orientation]:
            x, y = state.x + dx, state.y + dy
            self.board[y, x] = 1
        self.board, lines_cleared = self.clear_full_rows(self.board)
        self.total_lines_cleared += lines_cleared
        
    def get_updated_board(self, state) -> (np.ndarray, int, int):
        # Returns the board, the number of lines cleared, and the lock height, as if the
        # state was placed on the board but without actually placing the state
        board = self.board.copy()
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

        return board, lines_cleared