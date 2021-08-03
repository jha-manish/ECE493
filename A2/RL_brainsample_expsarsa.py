import numpy as np
import pandas as pd


class rlalgorithm:

    def __init__(self, actions, lr = 0.01, gamma = 0.9, epsilon = 0.1):
        self.actions = actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name = "ExpectedSarsa"

    def choose_action(self, observation):

        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_curr = self.q_table.loc[s, a]
        a_ = self.choose_action(str(s_))

        if s_ != 'terminal':
            q_new = r + self.gamma * self.get_expected(s_)
        else:
            q_new = r

        self.q_table.loc[s, a] += self.lr * (q_new - q_curr)

        return s_, a_

    def get_expected(self, s_):
        max_val = self.q_table.loc[s_].max()
        non_greedy_prob = self.epsilon / len(self.actions)
        num_greedy_actions = 0
        expected_val = 0

        for idx, val in enumerate(self.q_table.loc[s_]):
            if val == max_val:
                num_greedy_actions += 1

        greedy_prob = ((1 - self.epsilon) / num_greedy_actions) + non_greedy_prob

        for idx, val in enumerate(self.q_table.loc[s_]):
            if val == max_val:
                expected_val += self.q_table.loc[s_, idx] * greedy_prob
            else:
                expected_val += self.q_table.loc[s_, idx] * non_greedy_prob

        return expected_val


    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
