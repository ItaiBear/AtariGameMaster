"""An OpenAI Gym environment for Tetris."""
import os
import numpy as np
from nes_py import NESEnv
from typing import Callable

# the directory that houses this module
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

EMPTY = 239

# the table for looking up piece orientations
_PIECE_ORIENTATION_TABLE = [
    'Tu',
    'Tr',
    'Td',
    'Tl',
    'Jl',
    'Ju',
    'Jr',
    'Jd',
    'Zh',
    'Zv',
    'O',
    'Sh',
    'Sv',
    'Lr',
    'Ld',
    'Ll',
    'Lu',
    'Iv',
    'Ih',
]

# https://meatfighter.com/nintendotetrisai/#Representing_Tetriminos
_PIECE_SPAWN_ORIENTATION_ID = {
    'T' : 2,    # Td
    'J' : 7,    # Jd
    'Z' : 8,    # Zh
    'O' : 10,   # O
    'S' : 11,   # Sh
    'L' : 14,   # Ld
    'I' : 18,   # Ih
}


class TetrisEnv(NESEnv):
    """An environment for playing Tetris with OpenAI Gym."""

    # the legal range of rewards for each step
    reward_range = (-float('inf'), float('inf'))

    def __init__(self,
        b_type: bool = False,
        reward_score: bool = False,
        reward_lines: bool = True,
        penalize_height: bool = True,
        penalize_cost: bool = False,
        penalize_holes: bool = False,
        penalize_bumpiness: bool = False,
        penalize_transitions: bool = False,
        deterministic: bool = False,
        rom_name: str = 'Tetris.nes'
    ) -> None:
        """
        Initialize a new Tetris environment.

        Args:
            b_type: whether the game is A Type (false) or B Type (true)
            reward_score: whether to reward using the game score
            reward_lines: whether to reward using the number of lines cleared
            penalize_height: whether to penalize based on height of the board
            deterministic: true to disable RNG in the engine

        Returns:
            None

        """
        rom_path = os.path.join(_MODULE_DIR, '_roms', rom_name)
        super().__init__(rom_path)
        self._b_type = b_type

        self._reward_score = reward_score
        self._current_score = 0
        self._reward_lines = reward_lines
        self._current_lines = 0
        self._penalize_height = penalize_height
        self._current_height = 0
        self._penalize_cost = penalize_cost
        self._current_cost = 0
        self._penalize_holes = penalize_holes
        self._current_holes = 0
        self._penalize_bumpiness = penalize_bumpiness
        self._current_bumpiness = 0
        self._penalize_transitions = penalize_transitions
        self._current_transitions = 0

        self.deterministic = True  # Always use a deterministic starting point.
        # reset the emulator, skip the start screen, and backup the state
        self.reset()
        self._skip_start_screen()
        self._backup()
        self.reset()
        # Set the deterministic flag after setting up the engine.
        self.deterministic = deterministic

    @property
    def _is_game_over(self):
        """Return True if the game is over, False otherwise."""
        return bool(self.ram[0x0058])

    @property
    def _did_win_game(self):
        """Return True if game winning frame for B-type game mode."""
        if self._b_type:
            return self._number_of_lines == 0
        else: # can never win the A-type game
            return False

    def _read_bcd(self, address, length, little_endian=True):
        """
        Read a range of bytes where each nibble is a 10's place figure.

        Args:
            address: the address to read from as a 16 bit integer
            length: the number of sequential bytes to read
            little_endian: whether the bytes are in little endian order

        Returns:
            the integer value of the BCD representation

        """
        if little_endian:
            iterator = range(address, address + length)
        else:
            iterator = reversed(range(address, address + length))
        # iterate over the addresses to accumulate
        value = 0
        for idx, ram_idx in enumerate(iterator):
            value += 10**(2 * idx + 1) * (self.ram[ram_idx] >> 4)
            value += 10**(2 * idx) * (0x0F & self.ram[ram_idx])

        return value
    
    # MARK: RAM Hacks

    def _skip_start_screen(self):
        """Press and release start to skip the start screen."""
        # generate a random number for the Tetris RNG
        seed = 0, 0
        if not self.deterministic:
            seed = self.np_random.randint(0, 255), self.np_random.randint(0, 255)
        # seed = self.np_random.randint(0, 255), self.np_random.randint(0, 255)
        # skip garbage screens
        while self.ram[0x00C0] in {0, 1, 2, 3}:
            # seed the random number generator
            self.ram[0x0017:0x0019] = seed
            self._frame_advance(8)
            if self._b_type:
                self._frame_advance(128)
            self._frame_advance(0)

    # MARK: nes-py API calls

    def _did_reset(self):
        """Handle any RAM hacking after a reset occurs."""
        # skip frames and seed the random number generator
        seed = 0, 0
        if not self.deterministic:
            seed = self.np_random.randint(0, 255), self.np_random.randint(0, 255)
        for _ in range(14):
            self.ram[0x0017:0x0019] = seed
            self._frame_advance(0)

        # reset local variables
        self._current_score = 0
        self._current_lines = 0
        self._current_height = 0
        self._current_cost = 0
        self._current_holes = 0
        self._current_bumpiness = 0
        self._current_transitions = 0

    # MARK: Memory access

    def get_board(self):
        """Return the Tetris board from NES RAM."""
        return self.ram[0x0400:0x04C8].reshape((20, 10)).copy()

    def get_current_piece(self):
        """Return the current piece."""
        try:
            return _PIECE_ORIENTATION_TABLE[self.ram[0x0042]]
        except IndexError:
            return None

    def get_lines_being_cleared(self):
        """Return the number of cleared lines."""
        return self.ram[0x0056]

    def get_next_piece(self):
        """Return the current piece."""
        try:
            return _PIECE_ORIENTATION_TABLE[self.ram[0x00BF]]
        except IndexError:
            return None

    def get_statistics(self):
        """Return the statistics for the Tetrominoes."""
        return {
            'T': self._read_bcd(0x03F0, 2),
            'J': self._read_bcd(0x03F2, 2),
            'Z': self._read_bcd(0x03F4, 2),
            'O': self._read_bcd(0x03F6, 2),
            'S': self._read_bcd(0x03F8, 2),
            'L': self._read_bcd(0x03FA, 2),
            'I': self._read_bcd(0x03FC, 2),
        }
    
    def get_piece_count(self):
        """Return the current phase of the game"""
        piece_counts = self._statistics.values()
        return np.sum(list(piece_counts))

    def get_score(self):
        """Return the current score."""
        return self._read_bcd(0x0053, 3)

    def get_number_of_lines(self):
        """Return the number of cleared lines."""
        return self._read_bcd(0x0050, 2)

    def get_board_height(self):
        """Return the height of the board."""
        board = self._board
        # set the sentinel value for "empty" to 0
        board[board == EMPTY] = 0
        # look for any piece in any row
        board = board.any(axis=1)
        # take to sum to determine the height of the board
        return board.sum()
    
    def get_cost(self, board):
        """Return a cost for the current board state."""
        rows, cols = board.shape
        counts = np.count_nonzero(board, axis=1)
        assert counts.size == rows
        empty_counts = 10 - counts
        empty_counts[empty_counts == 10] = 0
        # empty_counts should contain the number of empty squares in each row with at least one full square, top-to-bottom
        row_costs = (np.arange(empty_counts.size)+1) / 20.0
        cost = np.dot(empty_counts, row_costs)
        #cost = np.sum(counts)
        return cost

    def get_hole_count(self, board):
        rows, cols = board.shape

        # Find the skyline for each column
        skyline = np.argmax(board, axis=0)
        skyline[np.all(board == 0, axis=0)] = rows

        # Create a mask for empty cells
        empty_cells = board == 0

        # Check if empty cells are above or at the skyline
        above_skyline = np.less.outer(np.arange(rows), skyline)

        # Check if empty cells are accessible from the left or right
        left_accessible = np.hstack([np.ones((rows, 1), dtype=bool), empty_cells[:, :-1]])
        right_accessible = np.hstack([empty_cells[:, 1:], np.ones((rows, 1), dtype=bool)])

        # Combine all accessibility conditions
        accessible = above_skyline | left_accessible | right_accessible

        # Count holes by selecting only non-accessible empty cells
        hole_count = np.sum(empty_cells & (1-accessible))

        return hole_count
    
    # Bumpiness measures the variance in consecutive column heights
    def get_bumpiness(self, board):
        rows, cols = board.shape

        # Find the skyline for each column
        skyline = np.argmax(board, axis=0)
        skyline[np.all(board == 0, axis=0)] = rows

        # Calculate bumpiness by summing the base-2 exponentials of height differences
        height_diffs = np.square(np.diff(skyline))
        height_diffs[height_diffs < 9] = 0
        # bumpiness = np.sum(np.square(height_diffs)) / 2
        bumpiness = np.mean(height_diffs)
        return bumpiness
    
    def get_transitions(self, board):
        # Count horizontal transitions
        horizontal_transitions = np.sum(np.abs(np.diff(board, axis=1)))

        # Count vertical transitions
        vertical_transitions = np.sum(np.abs(np.diff(board, axis=0)))

        # Combine horizontal and vertical transitions
        total_transitions = horizontal_transitions + vertical_transitions
        return total_transitions

    def _get_reward(self):
        """Return the reward after a step occurs."""
        board = self.get_board()
        # set the sentinel value for "empty" to 0
        board[board == EMPTY] = 0

        reward = 0

        if self._reward_score:
            new_score = self.get_score()
            reward += new_score - self._current_score
            self._current_score = new_score

        if self._reward_lines:
            new_lines = self.get_number_of_lines()
            reward += new_lines - self._current_lines
            self._current_lines = new_lines

        if self._penalize_height:
            new_height = self.get_board_height()
            reward -= new_height - self._current_height
            self._current_height = new_height

        if self._penalize_holes:
            new_holes = self.get_hole_count(board)
            reward -= new_holes - self._current_holes
            self._current_holes = new_holes

        if self._penalize_bumpiness:
            new_bumpiness = self.get_bumpiness(board)
            reward -= new_bumpiness - self._current_bumpiness
            self._current_bumpiness = new_bumpiness
        
        if self._penalize_transitions:
            new_transitions = self.get_transitions(board)
            reward -= new_transitions - self._current_transitions
            self._current_transitions = new_transitions
        
        if self._penalize_cost:
            new_cost = self.get_cost(board)
            reward -= new_cost - self._current_cost
            self._current_cost = new_cost

        return reward

    def _get_done(self):
        """Return True if the episode is over, False otherwise."""
        return self._is_game_over or self._did_win_game

    def _get_info(self):
        """Return the info after a step occurs."""
        return dict(
            score=self._score,
            piece_count=self._piece_count,
        )
            # current_piece=self._current_piece,
            # number_of_lines=self._number_of_lines,
            # next_piece=self._next_piece,
            # statistics=self._statistics,
            # board_height=self._board_height,
            # board=self._board,
            # piece_id=self.ram[0x00BF]


# explicitly define the outward facing API of this module
__all__ = [TetrisEnv.__name__]
