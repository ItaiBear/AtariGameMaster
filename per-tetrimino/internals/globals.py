

# Mapping of tetrimino IDs to their orientations
tetriminos = {
    "T": ["Tu", "Tr", "Td", "Tl"],
    "J": ["Jl", "Ju", "Jr", "Jd"],
    "Z": ["Zh", "Zv"],
    "O": ["O"],
    "S": ["Sh", "Sv"],
    "L": ["Lr", "Ld", "Ll", "Lu"],
    "I": ["Iv", "Ih"],
}

# Mapping of orientation IDs to their coordinates
orientations = { # (horizontal, vertical)
    "Tu": [(-1, 0), (0, 0), (1, 0), (0, -1)],
    "Tr": [(0, -1), (0, 0), (1, 0), (0, 1)],
    "Td": [(-1, 0), (0, 0), (1, 0), (0, 1)],
    "Tl": [(0, -1), (-1, 0), (0, 0), (0, 1)],
    
    "Jl": [(0, -1), (0, 0), (-1, 1), (0, 1)],
    "Ju": [(-1, -1), (-1, 0), (0, 0), (1, 0)],
    "Jr": [(0, -1), (1, -1), (0, 0), (0, 1)],
    "Jd": [(-1, 0), (0, 0), (1, 0), (1, 1)],
    
    "Zh": [(-1, 0), (0, 0), (0, 1), (1, 1)],
    "Zv": [(1, -1), (0, 0), (1, 0), (0, 1)],
    
    "O":  [(-1, 0), (0, 0), (-1, 1), (0, 1)],  # O doesn't rotate, so one entry is sufficient
    
    "Sh": [(0, 0), (1, 0), (-1, 1), (0, 1)],
    "Sv": [(0, -1), (0, 0), (1, 0), (1, 1)],
    
    "Lr": [(0, -1), (0, 0), (0, 1), (1, 1)],
    "Ld": [(-1, 0), (0, 0), (1, 0), (-1, 1)],
    "Ll": [(-1, -1), (0, -1), (0, 0), (0, 1)],
    "Lu": [(1, -1), (-1, 0), (0, 0), (1, 0)],
    
    "Iv": [(0, -2), (0, -1), (0, 0), (0, 1)],
    "Ih": [(-2, 0), (-1, 0), (0, 0), (1, 0)],
}

# Mapping of tetrimino IDs to their spawn orientations
spawn_orientations = {
    "T": "Td",
    "J": "Jd",
    "Z": "Zh",
    "O": "O",
    "S": "Sh",
    "L": "Ld",
    "I": "Ih",
}

shift_actions = ['left', 'right']
rotation_actions = ['clockwise', 'counterclockwise']

# Mapping of level to frames per drop
def frames_per_drop(level: int) -> int:
    level_to_frames = {0: 48, 1: 43, 2: 38, 3: 33, 4: 28,
                        5: 23, 6: 18, 7: 13, 8: 8, 9: 6}
    
    if level in level_to_frames:
        return level_to_frames[level]
    elif 10 <= level <= 12:
        return 5
    elif 13 <= level <= 15:
        return 4
    elif 16 <= level <= 18:
        return 3
    elif 19 <= level <= 28:
        return 2
    else:
        return 1