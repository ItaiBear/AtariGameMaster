from .globals import tetriminos, spawn_orientations, frames_per_drop
from .tetris import Tetris
from .state import State
from .evaluator import Evaluator
from collections import deque
import numpy as np


class Player:
    def __init__(self, board=None):
        if board is None:
            board = np.zeros((20, 10), dtype=np.int8)
        self.tetris: Tetris = Tetris(board)
        self.evaluator: Evaluator = Evaluator()
        self.visited_states = None
        
    def did_visit_state(self, state: State) -> bool:
        orientation_idx = tetriminos[state.piece].index(state.orientation)
        if self.visited_states[state.x, state.y, orientation_idx]: # Already visited
            #print(f"Already visited at state: {state.x}, {state.y}, {state.piece}, {state.orientation}")
            return True
        self.visited_states[state.x, state.y, orientation_idx] = True
        return False
        
    def bfs(self, tetrimino, fall_timer=0) -> list[State]:
        self.visited_states = np.zeros((self.tetris.width, self.tetris.height, 4), dtype=bool)
        spawn_state = State(5, 0, tetrimino, spawn_orientations[tetrimino], fall_timer=fall_timer)
        #print(f"Spawn state: {spawn_state.x}, {spawn_state.y}, {spawn_state.piece}, {spawn_state.orientation}")
        if not self.tetris.is_valid_state(spawn_state):
            # Game over
            #print("Game over")
            return []
        locked_states = []
        states_queue = deque([spawn_state])
        while states_queue:
            state = states_queue.popleft()
            
            left_states = state.left()
            right_states = state.right()
            down_states = state.down()
            clockwise_states = state.clockwise()
            counterclockwise_states = state.counterclockwise()

            
            all_valid = True
            for s in reversed(left_states):
                if not self.tetris.is_valid_state(s):
                    all_valid = False
                    break
            if all_valid and not self.did_visit_state(left_states[-1]):
                states_queue.append(left_states[-1])
                
            all_valid = True
            for s in reversed(right_states):
                if not self.tetris.is_valid_state(s):
                    all_valid = False
                    break
            if all_valid and not self.did_visit_state(right_states[-1]):
                states_queue.append(right_states[-1])
                
            all_valid = True
            for s in reversed(clockwise_states):
                if not self.tetris.is_valid_state(s):
                    all_valid = False
                    break
            if all_valid and not self.did_visit_state(clockwise_states[-1]):
                states_queue.append(clockwise_states[-1])
                
            all_valid = True
            for s in reversed(counterclockwise_states):
                if not self.tetris.is_valid_state(s):
                    all_valid = False
                    break
            if all_valid and not self.did_visit_state(counterclockwise_states[-1]):
                states_queue.append(counterclockwise_states[-1])
                
            all_valid = True
            for s in reversed(down_states):
                if not self.tetris.is_valid_state(s):
                    all_valid = False
                    break
            if all_valid and not self.did_visit_state(down_states[-1]):
                states_queue.append(down_states[-1])
            if not all_valid:
                locked_states.append(state)
                
        return locked_states
    
    def find_best_state(self, tetrimino, fall_timer=0) -> tuple[State, float, bool]:
        locked_states = self.bfs(tetrimino, fall_timer)
        #print(f"locked states: {len(locked_states)}")

        if not locked_states:
            return None, None, True
        
        best_state = None
        best_score = -np.inf
        for state in locked_states:
            board, lines_cleared, lock_height = self.tetris.get_updated_board(state)
            score = self.evaluator.evaluate(board, lines_cleared, lock_height)
            if score > best_score:
                best_state = state
                best_score = score
        return best_state, best_score, False
    
    def set_level(self, level: int) -> None:
        State.frames_per_drop = frames_per_drop(level)
        
    def get_current_board(self) -> np.ndarray:
        return self.tetris.board.copy()