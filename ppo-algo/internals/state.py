from .globals import tetriminos

class State:
    frames_per_drop = 10
    
    def __init__(self, x, y, piece, orientation, action=None, predecessor=None, fall_timer=0, auto_repeat=0):
        assert piece in tetriminos.keys(), f"piece {piece} not in {tetriminos.keys()}"
        assert orientation in tetriminos[piece], f"orientation {orientation} not in {tetriminos[piece]}"
        self.x: int = x
        self.y: int = y
        self.piece: str = piece
        self.orientation: str = orientation
        self.action: str = action
        self.predecessor: State = predecessor
        self.fall_timer: int = fall_timer # the fall timer after the action is taken
        self.auto_repeat: int = auto_repeat
        self.drop: bool = False
        self.down_startup: bool = True
        #print(f"frames per drop: {State.frames_per_drop}")
        
    def get_should_drop(self):
        fall_timer = self.fall_timer + 1
        
        if not self.action == 'down':
            self.down_startup = True
        
        if self.action == 'down' and self.drop:
            if self.down_startup:
                return False, 0
            return False, fall_timer
        
        self.drop = fall_timer >= State.frames_per_drop
        fall_timer = (not self.drop)*fall_timer
        return self.drop, fall_timer
        
    def noop(self) -> 'State':
        drop, next_fall_timer = self.get_should_drop()
        noop_state = State(self.x, self.y+drop, self.piece, self.orientation, 'noop', self, next_fall_timer)
        return noop_state
    
    def left(self) -> list['State']:
        # actions: noop, left
        states = []
        if self.action == 'left':
            self = self.noop()
            states.append(self)
        drop, next_fall_timer = self.get_should_drop()
        states.append(State(self.x-1, self.y+drop, self.piece, self.orientation, 'left', self, next_fall_timer))
        if self.action == 'down':
            states.append(states[-1].noop())
        return states
    
    def right(self) -> list['State']:
        # actions: noop, right
        states = []
        if self.action == 'right':
            self = self.noop()
            states.append(self)
        drop, next_fall_timer = self.get_should_drop()
        states.append(State(self.x+1, self.y+drop, self.piece, self.orientation, 'right', self, next_fall_timer))
        if self.action == 'down':
            states.append(states[-1].noop())
        return states

    def down(self) -> list['State']:
        drop, next_fall_timer = self.get_should_drop()
        if self.auto_repeat >= 2:
            down_state = State(self.x, self.y+1, self.piece, self.orientation, 'down', self, 0, auto_repeat=1)
            down_state.drop = True
            down_state.down_startup = False
            return [down_state]
        else:
            down_state = State(self.x, self.y+drop, self.piece, self.orientation, 'down', self, next_fall_timer, auto_repeat=self.auto_repeat+1)
            down_states = down_state.down()
            return [down_state] + down_states
    
    def clockwise(self) -> list['State']:
        states = []
        if self.action == 'clockwise':
            self = self.noop()
            states.append(self)
        drop, next_fall_timer = self.get_should_drop()
        orientation = tetriminos[self.piece][(tetriminos[self.piece].index(self.orientation) + 1) % len(tetriminos[self.piece])]
        states.append(State(self.x, self.y+drop, self.piece, orientation, 'clockwise', self, next_fall_timer))
        return states
    
    def counterclockwise(self) -> list['State']:
        states = []
        if self.action == 'counterclockwise':
            self = self.noop()
            states.append(self)
        drop, next_fall_timer = self.get_should_drop()
        orientation = tetriminos[self.piece][(tetriminos[self.piece].index(self.orientation) - 1) % len(tetriminos[self.piece])]
        states.append(State(self.x, self.y+drop, self.piece, orientation, 'counterclockwise', self, next_fall_timer))
        return states
    
    def get_action_sequence(self) -> list[str]:
        actions = []
        state = self
        while state.predecessor:
            actions.append(state.action)
            state = state.predecessor
        return actions[::-1]
    
    def get_state_sequence(self) -> list['State']:
        states = []
        state = self
        while state.predecessor:
            states.append(state)
            state = state.predecessor
        return states[::-1]