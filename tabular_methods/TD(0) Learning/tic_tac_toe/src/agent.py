from ttk_env import Env
import numpy as np
from typing import Tuple, Union


class Agent:
    """
    Tabular Agent implementing Temporal Difference (TD) learning.
    Updates value estimates based on the immediate next state.
    """
    def __init__(self, symbol, alpha=0.1, epsilon=0.1):
        self.symbol = symbol
        self.alpha = alpha
        self.epsilon = epsilon
        self.values = {}
        self.state_history = []

    def get_value(self, state: Tuple) -> float:
        board = Env.to_array(state)
        winner = Env.check_winner(board)

        if winner is not None:
            if winner == self.symbol:
                value = 1.0
            elif winner == 0:
                value = 0.5
            else:
                value = 0.0
            return value

        if state in self.values:
            return self.values[state]

        return 0.5

    def choose_action(self, state: Tuple) -> Union[Tuple, bool]:
        board = Env.to_array(state)
        possible_moves = Env.get_possible_moves(board)

        if np.random.rand() < self.epsilon:
            return possible_moves[np.random.choice(len(possible_moves))], True

        best_value = -1
        best_move = None

        for move in possible_moves:
            possible_board = board.copy()
            possible_board[move] = self.symbol
            possible_state = Env.to_tuple(possible_board)
            state_value = self.get_value(possible_state)

            if state_value >= best_value:
                best_value = state_value
                best_move = move

        return best_move, False

    def update_history(self, state: Tuple, was_exploratory: bool) -> None:
        self.state_history.append((state, was_exploratory))

    def learn(self):
        for i in range(len(self.state_history) - 2, -1, -1):
            current_state, _ = self.state_history[i]
            next_state, next_state_was_exploratory = self.state_history[i + 1]
            if next_state_was_exploratory:
                continue
            updated_value = self.get_value(current_state) + self.alpha * (
                self.get_value(next_state) - self.get_value(current_state)
            )
            self.values[current_state] = updated_value

        self.state_history = []
