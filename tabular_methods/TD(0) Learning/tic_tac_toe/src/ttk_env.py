import numpy as np
from typing import Tuple

input_to_state_map = {
    "A": (0, 0),
    "Z": (0, 1),
    "E": (0, 2),
    "Q": (1, 0),
    "S": (1, 1),
    "D": (1, 2),
    "W": (2, 0),
    "X": (2, 1),
    "C": (2, 2),
}

state_to_input_map = {v: k for k, v in input_to_state_map.items()}
symbol_map = {1: "X", -1: "O", 0: " "}

N_COLS = N_ROWS = 3


class Env:
    def __init__(self):
        self.reset()

    @staticmethod
    def to_tuple(state : np.array) -> Tuple:
        return tuple(int(x) for x in state.flatten())

    @staticmethod
    def to_array(state : Tuple) -> np.array:
        return np.array(state).reshape(N_ROWS, N_COLS)

    def reset(self):
        self.board = np.zeros((N_ROWS, N_COLS))
        self.winner = None
        self.current_player = 1
        self.done = False
        return self.to_tuple(self.board)

    @staticmethod
    def check_winner(state : Tuple):
        board = Env.to_array(state)

        for i in range(N_ROWS):
            if abs(sum(board[i, :])) == N_COLS:
                return board[i, 0]

        for j in range(N_COLS):
            if abs(sum(board[:, j])) == N_ROWS:
                return board[0, j]

        if abs(np.trace(board)) == N_ROWS:
            return board[0, 0]
        if abs(np.trace(np.fliplr(board))) == N_ROWS:
            return board[0, N_COLS - 1]

        if len(Env.get_possible_moves(board)) == 0:
            return 0

        return None

    @staticmethod
    def get_possible_moves(state : Tuple):
        board = Env.to_array(state)
        moves = []
        for i in range(N_ROWS):
            for j in range(N_COLS):
                if board[i, j] == 0:
                    moves.append((i, j))
        return moves

    def step(self, action : Tuple):
        if self.board[action] != 0:
            raise ValueError("Illegal move, try again")

        self.board[action] = self.current_player

        winner = self.check_winner(self.board)

        reward = 0
        done = False

        if winner is not None:
            self.winner = winner
            done = True
            if winner == 1:
                reward = 1.0
            elif winner == -1:
                reward = 0.0
            elif winner == 0:
                reward = 0.5
        else:
            done = False
            reward = 0
            self.current_player *= -1

        next_state = self.to_tuple(self.board)
        return (next_state, reward, done)

        
    def render(self):
        RED = "\033[91m"
        GREEN = "\033[92m"
        GREY = "\033[90m"
        RESET = "\033[0m"

        print("")

        row_strings = []
        for r in range(N_ROWS):
            cell_strings = []
            for c in range(N_COLS):
                symbol = self.board[r, c]
                
                if symbol == 1:
                    content = f"{RED}X{RESET}" 
                elif symbol == -1:
                    content = f"{GREEN}O{RESET}"
                else:
                    key = state_to_input_map.get((r, c), " ")
                    content = f"{GREY}{key}{RESET}"
                
                cell_strings.append(f" {content} ")
            
            row_strings.append("|".join(cell_strings))

        separator = "\n---+---+---\n"
        print(separator.join(row_strings))
        print("")

  