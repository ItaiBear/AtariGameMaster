

def format_board(board) -> str:
    # formats board to be human readable
    board_str = ""
    for row in board:
        for col in row:
            board_str += " " + str(int(col))
        board_str += "\n"
    return board_str  