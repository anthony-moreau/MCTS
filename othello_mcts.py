import numpy as np
from collections import defaultdict
import othello_rules as rules
import plotly.graph_objs as go


class MonteCarloTreeSearchNode:
    def __init__(self, state, simulation_no=100, c_param=0.1, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        # self._untried_actions = None
        self._untried_actions = self.untried_actions()
        self.simulation_no = simulation_no
        self.c_param = c_param

    def untried_actions(self):

        self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = self.state.move(action)
        child_node = MonteCarloTreeSearchNode(next_state, simulation_no=self.simulation_no, c_param=self.c_param,
                                              parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state

        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()

            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result()

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=None):
        if c_param is None:
            c_param = self.c_param
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):

        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):

        current_node = self
        while not current_node.is_terminal_node():

            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        for i in range(self.simulation_no):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        return self.best_child(c_param=0.)


def MCTS(initial_state, simulation_no=100, c_param = 0.01):
    root = MonteCarloTreeSearchNode(state=initial_state, simulation_no=simulation_no, c_param=c_param)
    selected_node = root.best_action()
    return selected_node


def simulate_game(simulation_no_J1=100, c_param_J1=0.01, random_side_1=False, simulation_no_J2=100, c_param_J2=0.01,
                  random_side_2 = False):
    game_state = rules.State()
    while not game_state.is_game_over():
        if game_state.side == 1:
            if random_side_1:
                possible_moves =game_state.get_legal_actions()
                game_state = game_state.move(possible_moves[np.random.randint(len(possible_moves))])
            else:
                game_state = game_state.move(
                    MCTS(game_state, simulation_no=simulation_no_J1, c_param=c_param_J1).parent_action)
        else:
            if random_side_2:
                possible_moves = game_state.get_legal_actions()
                game_state = game_state.move(possible_moves[np.random.randint(len(possible_moves))])
            else:
                game_state = game_state.move(
                    MCTS(game_state, simulation_no=simulation_no_J2, c_param=c_param_J2).parent_action)
    print(game_state)
    result = game_state.game_result()
    print(result)
    if result == 1:
        return game_state.master_side
    elif result == -1:
        return 3 - game_state.master_side
    else:
        return 0


simulation_amount = [1, 100]
n_repeat = 10

results = []

for c_param in [0.1, 1, 2]:
    print(f"current c parameter: {c_param}")
    results.append([simulate_game(c_param_J2=c_param) for _ in range(n_repeat)])

print(results)