
import numpy as np
import pandas as pd


class rlalgorithm:
    def __init__(self, actions, lr = 0.01, gamma = 0.9, epsilon = 0.1):
        self.actions = actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name = 'QLearning'

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
        a_ = self.choose_action(str(s_))

        if s_ != 'terminal':
            q = r + self.gamma * self.q_table.loc[s_].max()
        else:
            q = r

        self.q_table.loc[s, a] += self.lr * (q - self.q_table.loc[s, a])

        return s_, a_

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
