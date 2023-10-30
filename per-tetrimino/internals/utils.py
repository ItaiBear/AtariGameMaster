

def format_board(board) -> str:
    board_str = ""
    for row in board:
        for col in row:
            board_str += " " + str(int(col))
        board_str += "\n"
    return board_str  