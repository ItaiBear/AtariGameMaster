"""An OpenAI Gym environment for Tetris."""
import os
import numpy as np
from nes_py import NESEnv
from typing import Callable

# the directory that houses this module
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

EMPTY = 239

BUTTON_MAP = {
    'right':  0b10000000,
    'left':   0b01000000,
    'down':   0b00100000,
    'up':     0b00010000,
    'start':  0b00001000,
    'select': 0b00000100,
    'B':      0b00000010,
    'A':      0b00000001,
    'NOOP':   0b00000000,
}

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

BOARD_ROWS, BOARD_COLS = 20,10
NUM_METRICS = 7
LINES, HEIGHT, COST, HOLES, BUMPINESS, COL_TRANSITIONS, ROW_TRANSITIONS = range(NUM_METRICS)

class TetrisEnv(NESEnv):
    """An environment for playing Tetris with OpenAI Gym."""

    # the legal range of rewards for each step
    reward_range = (-float('inf'), float('inf'))

    def __init__(self,
        b_type: bool = False,
        # score_weight: float = 1,
        line_weight: float = 1,
        height_weight: float = -1,
        cost_weight: float = -1,
        holes_weight: float = -1,
        bumpiness_weight: float = -1,
        col_transitions_weight: float = -1,
        row_transitions_weight: float = -1,
        deterministic: bool = False,
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
        rom_path = os.path.join(_MODULE_DIR, '_roms', 'Tetris.nes')
        super().__init__(rom_path)
        self._b_type = b_type

        self._weights = np.array([line_weight,
                                  height_weight,
                                  cost_weight,
                                  holes_weight,
                                  bumpiness_weight,
                                  col_transitions_weight,
                                  row_transitions_weight])
        assert NUM_METRICS == self._weights.size
        # Get metrics for an empty board
        self._metrics = self.get_metrics(np.zeros((BOARD_ROWS, BOARD_COLS)))

        self.deterministic = True  # Always use a deterministic starting point.
        # reset the emulator, skip the start screen, and backup the state
        self.reset()
        self._skip_start_screen(level_9=True)
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

    def _skip_start_screen(self, level_9 : bool = False):
        """Press and release start to skip the start screen."""
        # generate a random number for the Tetris RNG
        seed = 0, 0
        if not self.deterministic:
            seed = self.np_random.randint(0, 255), self.np_random.randint(0, 255)
        # seed = self.np_random.randint(0, 255), self.np_random.randint(0, 255)

        assert level_9 and not self._b_type

        # skip garbage screens
        while self.ram[0x00C0] in {0, 1, 2, 3}:
            state = self.ram[0x00C0]
            if state in {0,1,2}:
                # Opening Screen, Title screen, Game-Mode Screen
                self._frame_advance(BUTTON_MAP["NOOP"])
                self._frame_advance(BUTTON_MAP["start"])
                self._frame_advance(BUTTON_MAP["NOOP"])
            else:
                # Level Select - select level 9
                for _ in range(2):
                    # Select bottom right option (level nine)
                    self._frame_advance(BUTTON_MAP["NOOP"])
                    for __ in range(8):
                        self._frame_advance(BUTTON_MAP["right"])
                        self._frame_advance(BUTTON_MAP["NOOP"])
                    self._frame_advance(BUTTON_MAP["down"])
                    self._frame_advance(BUTTON_MAP["NOOP"])
                # Start game
                self._frame_advance(BUTTON_MAP["NOOP"])
                self._frame_advance(BUTTON_MAP["start"])
                self._frame_advance(BUTTON_MAP["NOOP"])

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

        # reset metrics
        self._metrics = self.get_metrics(np.zeros((BOARD_ROWS, BOARD_COLS)))

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
        piece_counts = self.get_statistics().values()
        return np.sum(list(piece_counts))
    
    # Reward Metrics
    def get_score(self):
        """Return the current score."""
        return self._read_bcd(0x0053, 3)

    def get_number_of_lines(self):
        """Return the number of cleared lines."""
        return self._read_bcd(0x0050, 2)

    def get_board_height(self, skyline):
        """Return the height of the board."""
        if self._weights[HEIGHT] < 1e-8:
            return 0
        height = BOARD_ROWS - np.min(skyline)
        # Normalize to keep between 0 and 1
        return height / 20
    
    def get_cost(self, board):
        """Return a cost for the current board state."""
        if abs(self._weights[COST]) < 1e-8:
            return 0
        # Number of solid cells in each rows
        counts = np.count_nonzero(board, axis=1)
        empty_counts = BOARD_COLS - counts
        empty_counts[empty_counts == BOARD_COLS] = 0
        # empty_counts should contain the number of empty squares in each row with at least one full square, top-to-bottom
        row_costs = np.arange(empty_counts.size)+1
        cost = np.dot(empty_counts, row_costs)
        # Normalize to keep (mostly) between 0 and 1
        return cost / 1600

    def get_hole_count(self, board, skyline):
        """"Return the number of holes in the board"""
        if abs(self._weights[HOLES]) < 1e-8:
            return 0
        # Create a 2D boolean array that is True wherever cells are empty
        empty_cells = (board == 0)
        # Create a 2D boolean array that is True anywhere below the skyline
        below_skyline = np.greater.outer(np.arange(BOARD_ROWS), skyline)
        # A hole is any empty cell below the skyline
        hole_count = np.sum(np.logical_and(empty_cells, below_skyline))
        # Normalize to keep (mostly) between 0 and 1
        return hole_count / 100
    
    # Bumpiness measures the variance in consecutive column heights
    def get_bumpiness(self, skyline):
        if abs(self._weights[BUMPINESS]) < 1e-8:
            return 0
        # Calculate bumpiness by summing the base-2 exponentials of height differences
        height_diffs = np.square(np.diff(skyline))
        # Bumpiness is the average squared height difference
        bumpiness = np.mean(height_diffs)
        return bumpiness / 100
    
    def get_col_transitions(self, board):
        """"Return the number of column transitions"""
        # A column transition is a switch between a solid cell and an empty cell along the same column
        return np.sum(np.abs(np.diff(board, axis=0))) / 20
    
    def get_row_transitions(self, board):
        # A column transition is a switch between a solid cell and an empty cell along the same column
        return np.sum(np.abs(np.diff(board, axis=1))) / 20

    def get_metrics(self, board):
        # Useful data that many metrics need
        skyline = np.argmax(board, axis=0)
        skyline[np.logical_and(skyline == 0, board[0] == 0)] = BOARD_ROWS

        # Penalize every reward except number of lines by making them negative
        metrics = np.array([+self.get_number_of_lines(),
                            -self.get_board_height(skyline),
                            -self.get_cost(board),
                            -self.get_hole_count(board, skyline),
                            -self.get_bumpiness(skyline),
                            -self.get_col_transitions(board),
                            -self.get_row_transitions(board)])
        assert NUM_METRICS == metrics.size
        return metrics

    def _get_reward(self):
        """Return the reward after a step occurs."""
        board = self.get_board()
        # Set empty cells to 0 and full cells to 1
        board = np.where(board == EMPTY, 0, 1)
        assert board.shape == (BOARD_ROWS, BOARD_COLS)

        new_metrics = self.get_metrics(board)
        reward = np.dot(self._weights, new_metrics - self._metrics)
        self._metrics = new_metrics

        return reward

    def _get_done(self):
        """Return True if the episode is over, False otherwise."""
        return self._is_game_over or self._did_win_game

    def _get_info(self):
        """Return the info after a step occurs."""
        return dict(
            # score=self.get_score(),
            piece_count=self.get_piece_count(),
            lines=self.get_number_of_lines(),
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
