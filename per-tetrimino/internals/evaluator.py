import numpy as np


class Evaluator:
    def __init__(self, weights=None):
        if weights is None:
            self.weights = {
                'Total Lines Cleared': 1.0,
                'Total Lock Height': 12.88,
                'Total Well Cells': 15.84,
                'Total Column Holes': 26.89,
                'Total Column Transitions': 27.61,
                'Total Row Transitions': 30.18
            }
        else:
            self.weights = weights

    def total_lines_cleared(self, lines_cleared):
        """
        Total lines cleared
        """
        return lines_cleared

    
    def total_lock_height(self, lock_height):
        """
        Lock height is the height above the playfield floor where a piece locked. 
        It is the vertical distance that a locked piece would drop if all the other 
        solid squares in the playfield were removed and the orientation of the piece
        was maintained.
        """
        #print(f"Lock height: {lock_height}")
        return -lock_height

    def total_well_cells(self, board):
        """
        A well cell is an empty cell located above all the solid cells within its column
        such that its left and right neighbors are both solid cells; the playfield walls
        are treated as solid cells in this determination. The idea is that a well
        is a structure open at the top, sealed at the bottom and surrounded by walls
        on both sides. The possibility of intermittent gaps in the well walls means that
        well cells do not necessarily appear in a contiguous stack within a column.
        """
        skyline = np.argmax(board, axis=0)

        well_cells = 0
        for col in range(board.shape[1]):
            for row in range(skyline[col], board.shape[0]):
                if board[row, col] == 0:
                    if col == 0:
                        if board[row, col+1] == 1:
                            well_cells += 1
                    elif col == board.shape[1]-1:
                        if board[row, col-1] == 1:
                            well_cells += 1
                    else:
                        if board[row, col-1] == 1 and board[row, col+1] == 1:
                            well_cells += 1
        #print(f"Well cells: {well_cells}")
        return -well_cells

    def total_column_holes(self, board):
        """
        A column hole is an empty cell directly beneath a solid cell.
        The playfield floor is not compared to the cell directly above it.
        Empty columns contain no holes.
        """
        holes = 0
        for col in range(board.shape[1]):
            for row in range(1, board.shape[0]):
                if board[row, col] == 0 and board[row-1, col] == 1:
                    holes += 1
        #print(f"Holes: {holes}")
        return -holes

    def total_column_transitions(self, board):
        """
        A column transition is an empty cell adjacent to a solid cell (or vice versa)
            within the same column.
        The changeover from the highest solid block in the column to the empty space
            above it is not considered a transition. 
        Similarly, the playfield floor is not compared to the cell directly above it.
        As a result, a completely empty column has no transitions.
        """
        return -np.sum(np.diff(board, axis=0) != 0)

    # Total row transitions
    def total_row_transitions(self, board):
        """
        A row transition is an empty cell adjacent to a solid cell (or vice versa)
            within the same row.
        Empty cells adjoining playfield walls are considered transitions.
        The total is computed across all rows in the playfield.
        However, rows that are completely empty do not contribute to the sum.
        """
        return -np.sum(np.diff(board, axis=1) != 0)

    def evaluate(self, board, lines_cleared, lock_height):
        score = 0
        score += self.weights['Total Lines Cleared'] * self.total_lines_cleared(lines_cleared)
        score += self.weights['Total Lock Height'] * self.total_lock_height(lock_height)
        score += self.weights['Total Well Cells'] * self.total_well_cells(board)
        score += self.weights['Total Column Holes'] * self.total_column_holes(board)
        score += self.weights['Total Column Transitions'] * self.total_column_transitions(board)
        score += self.weights['Total Row Transitions'] * self.total_row_transitions(board)
        return score