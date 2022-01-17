import numpy as np
import copy as cp

DIRECTIONS = [np.array((0, 1)), np.array((1, 0)), np.array((1, 1)),
              np.array((0, -1)), np.array((-1, 0)), np.array((-1, -1)), np.array((1, -1)), np.array((-1, 1))]


class State:
    def __init__(self, grid=None, previous_skip=False, side=2, master_side=2):
        if grid is None:
            self.grid = [6 * [0] for i in range(6)]
            self.grid[2][2] = 1
            self.grid[2][3] = 2
            self.grid[3][2] = 2
            self.grid[3][3] = 1
        else:
            self.grid = grid
        self.side = side
        self.previous_skip = previous_skip
        self.master_side = master_side

    def get_legal_actions(self):
        '''
        Modify according to your game or
        needs. Constructs a list of all
        possible actions from current state.
        Returns a list.
        '''
        return get_side_moves(self.grid, self.side)

    def is_game_over(self):
        '''
        Modify according to your game or
        needs. It is the game over condition
        and depends on your game. Returns
        true or false
        '''
        return is_game_over(self.grid, self.side, self.previous_skip)

    def game_result(self):
        '''
        Modify according to your game or
        needs. Returns 1 or 0 or -1 depending
        on your state corresponding to win,
        tie or a loss.
        '''
        return get_winning_side(self.grid, self.master_side)

    def move(self, action):
        '''
        Modify according to your game or
        needs. Changes the state of your
        board with a new value. For a normal
        Tic Tac Toe game, it can be a 3 by 3
        array with all the elements of array
        being 0 initially. 0 means the board
        position is empty. If you place x in
        row 2 column 3, then it would be some
        thing like board[2][3] = 1, where 1
        represents that x is placed. Returns
        the new state after making a move.
        '''
        new_state = State(grid=cp.deepcopy(self.grid), previous_skip=self.previous_skip, side=self.side,
                          master_side=self.master_side)
        apply_move(new_state, action)
        new_state.side = 3 - self.side
        return new_state

    def __str__(self):
        string = f"Current player {self.side}\n"
        for line in self.grid:
            string += str(line) + "\n"
        return string

    def __repr__(self):
        return self.__str__()


def position_in_grid(grid, position):
    return 0 <= position[0] < len(grid) and 0 <= position[1] < len(grid[0])


def get_claimable_positions_from(grid: list[list[int]], source: np.ndarray):
    disk_side = grid[source[0]][source[1]]
    other_side = 3 - disk_side
    claimable_pos = []
    for vector in DIRECTIONS:
        new_pos = source + vector
        if position_in_grid(grid, new_pos) and grid[new_pos[0]][new_pos[1]] == other_side:
            while position_in_grid(grid, new_pos) and grid[new_pos[0]][new_pos[1]] == other_side:
                new_pos += vector
            if position_in_grid(grid, new_pos) and grid[new_pos[0]][new_pos[1]] == 0:
                claimable_pos.append(new_pos)
    return claimable_pos


def get_side_disks(grid, side):
    disks = []
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == side:
                disks.append(np.array((i, j)))
    return disks


def get_side_moves(grid, side):
    disks = get_side_disks(grid, side)
    moves = {}
    for coord in disks:
        disk_moves = get_claimable_positions_from(grid, coord)
        for move in disk_moves:
            non_mutable_coord = tuple(move)
            if non_mutable_coord not in moves:
                moves[non_mutable_coord] = [coord]
            else:
                moves[non_mutable_coord] += [coord]
    side_moves = [None]
    if len(moves) > 0:
        side_moves = [[move, moves[move]] for move in moves.keys()]
    return side_moves


def apply_move(state, move):
    if move is not None:
        state.previous_skip = False
        coord, list_position_from = move[0], move[1]
        state.grid[coord[0]][coord[1]] = state.side
        for position_from in list_position_from:
            direction = position_from - coord
            # getting the vector back to one unit for each direction
            if direction[0] > 0:
                direction[0] = 1
            elif direction[0] < 0:
                direction[0] = -1
            if direction[1] > 0:
                direction[1] = 1
            elif direction[1] < 0:
                direction[1] = -1
            new_pos = coord + direction
            while new_pos[0] != position_from[0] or new_pos[1] != position_from[1]:
                state.grid[new_pos[0]][new_pos[1]] = state.side
                new_pos += direction
    else:
        state.previous_skip = True


def explore_all_possible_games(state: State):
    moves = state.get_legal_actions()
    for move in moves:
        explore_all_possible_games(state.move(move))


def get_winning_side(grid, master_side):
    side_count = {1: 0, 2: 0}
    for line in grid:
        for tile in line:
            if tile:
                side_count[tile] += 1
    if side_count[1] == side_count[2]:
        return 0
    elif side_count[1] > side_count[2]:
        if master_side == 1:
            return 1
        else:
            return -1
    else:
        if master_side == 2:
            return 1
        else:
            return -1


def position_can_claim(grid: list[list[int]], source: np.ndarray):
    disk_side = grid[source[0]][source[1]]
    other_side = 3 - disk_side
    for vector in DIRECTIONS:
        new_pos = source + vector
        if position_in_grid(grid, new_pos) and grid[new_pos[0]][new_pos[1]] == other_side:
            while position_in_grid(grid, new_pos) and grid[new_pos[0]][new_pos[1]] == other_side:
                new_pos += vector
            if position_in_grid(grid, new_pos) and grid[new_pos[0]][new_pos[1]] == 0:
                return True
    return False


def is_game_over(grid, side, previous_skip):
    if previous_skip:
        disks = get_side_disks(grid, side)
        for coord in disks:
            if position_can_claim(grid, coord):
                return False
        return True
    return False

# Othello default grid
# grid = [8 * [0] for i in range(8)]
# grid[3][3] = 1
# grid[3][4] = 2
# grid[4][3] = 2
# grid[4][4] = 1

# Othello 6*6 grid
# grid = [6 * [0] for i in range(6)]
# grid[2][2] = 1
# grid[2][3] = 2
# grid[3][2] = 2
# grid[3][3] = 1

# faulty_grid = [[0, 0, 0, 0, 0, 0], [0, 2, 2, 2, 2, 0], [0, 2, 2, 2, 2, 2], [0, 2, 1, 1, 2, 2], [0, 2, 1, 1, 2, 2], [0, 2, 2, 2, 2, 2]]
#
# print(get_claimable_positions_from(faulty_grid, np.array((3,2))))
# print("hello")
